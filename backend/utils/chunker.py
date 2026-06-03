import os
import ast
import re

# ── Fallback chunker (sliding window) ────────────────────────────────────────

def fallback_chunking(file_path: str, code: str, chunk_size: int = 50) -> list:
    """Sliding-window chunking for files where structured parsing fails."""
    chunks = []
    lines = code.splitlines()
    if not lines:
        return []
    i = 0
    while i < len(lines):
        end = min(i + chunk_size, len(lines))
        chunks.append({
            "file_path": file_path,
            "start_line": i + 1,
            "end_line": end,
            "content": "\n".join(lines[i:end]),
            "name": f"lines {i + 1}-{end}",
            "type": "raw_lines",
        })
        i += max(1, chunk_size - 10)  # 10-line overlap
    return chunks


# ── Python AST chunker ────────────────────────────────────────────────────────

def chunk_python_file(file_path: str, code: str) -> list:
    """
    Split a Python file into semantic chunks using the stdlib AST.

    FIX #16: Always emit a module-header chunk (imports + constants at the top
    of the file) so retrieval can surface global configuration and imports.

    FIX #17: `visit_ClassDef` no longer calls `self.generic_visit(node)`,
    which previously caused every method inside a class to be emitted as a
    *second* chunk on top of the full-class chunk.  Top-level functions are
    still visited because the visitor traverses the module's direct children.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return fallback_chunking(file_path, code)

    lines = code.splitlines()
    num_lines = len(lines)
    chunks: list[dict] = []

    # ── FIX #16: Module-header chunk ──────────────────────────────────────
    # Collect the line number where the first top-level class/function starts.
    first_def_line = num_lines  # default: whole file is header
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.col_offset == 0:  # only top-level definitions
                first_def_line = min(first_def_line, node.lineno - 1)

    if first_def_line > 0:
        header_content = "\n".join(lines[:first_def_line])
        if header_content.strip():
            chunks.append({
                "file_path": file_path,
                "start_line": 1,
                "end_line": first_def_line,
                "content": header_content,
                "name": "module header",
                "type": "module_header",
            })

    # ── Visitor: classes and top-level functions ───────────────────────────
    class CodeVisitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef):
            start = node.lineno
            end = getattr(node, "end_lineno", num_lines)
            chunks.append({
                "file_path": file_path,
                "start_line": start,
                "end_line": end,
                "content": "\n".join(lines[start - 1 : end]),
                "name": f"class {node.name}",
                "type": "class",
            })
            # FIX #17: Do NOT call self.generic_visit(node) here.
            # Descending into the class would visit each method and emit it as
            # an additional, duplicate chunk (method content already included
            # in the full-class chunk above).

        def visit_FunctionDef(self, node: ast.FunctionDef):
            start = node.lineno
            end = getattr(node, "end_lineno", num_lines)
            chunks.append({
                "file_path": file_path,
                "start_line": start,
                "end_line": end,
                "content": "\n".join(lines[start - 1 : end]),
                "name": f"def {node.name}",
                "type": "function",
            })
            # Still descend so nested (inner) functions are also captured
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            start = node.lineno
            end = getattr(node, "end_lineno", num_lines)
            chunks.append({
                "file_path": file_path,
                "start_line": start,
                "end_line": end,
                "content": "\n".join(lines[start - 1 : end]),
                "name": f"async def {node.name}",
                "type": "function",
            })
            self.generic_visit(node)

    CodeVisitor().visit(tree)

    if not chunks:
        return fallback_chunking(file_path, code)
    return chunks


# ── Brace-language chunker ────────────────────────────────────────────────────

# Characters that open/close a string literal in JS/TS/Go/Java/C++.
# We track these to avoid counting braces inside strings.
_STRING_OPENERS = {'"', "'", "`"}


def _count_braces_in_line(line: str) -> tuple[int, int]:
    """
    Count `{` and `}` occurrences in *line*, skipping characters inside
    string literals and line comments.

    FIX #18: The previous implementation counted every `{` and `}` character
    regardless of context, so braces inside strings (`"{"`) or template
    literals (`` `${x}` ``) produced malformed chunks.
    """
    opens = 0
    closes = 0
    in_string: str | None = None  # the quote character that opened the string
    escape_next = False
    i = 0
    while i < len(line):
        ch = line[i]

        if escape_next:
            escape_next = False
            i += 1
            continue

        if ch == "\\":
            escape_next = True
            i += 1
            continue

        if in_string:
            if ch == in_string:
                in_string = None  # closing quote
        else:
            # Line comments: // … for JS/TS/Go/C++  — stop counting for this line
            if ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                break
            if ch in _STRING_OPENERS:
                in_string = ch
            elif ch == "{":
                opens += 1
            elif ch == "}":
                closes += 1

        i += 1

    return opens, closes


def chunk_brace_file(file_path: str, code: str) -> list:
    """
    Split brace-delimited source files (JS, TS, Go, Java, C++, JSX, TSX)
    by function/class block boundaries.
    """
    chunks: list[dict] = []
    lines = code.splitlines()
    n = len(lines)

    patterns = [
        r"\bclass\s+\w+",
        r"\bfunction\s+\w*",
        r"\bfunc\s+",
        r"\b\w+\s*\([^)]*\)\s*\{",
        r"\bconst\s+\w+\s*=\s*(?:\([^)]*\)|\w+)\s*=>",
    ]
    combined_pattern = re.compile("|".join(patterns))

    i = 0
    while i < n:
        if combined_pattern.search(lines[i]):
            start_line = i + 1
            found_open = False
            brace_count = 0
            block_lines: list[str] = []

            j = i
            while j < n:
                block_lines.append(lines[j])
                opens, closes = _count_braces_in_line(lines[j])

                if opens > 0:
                    found_open = True
                brace_count += opens - closes

                if found_open and brace_count <= 0:
                    end_line = j + 1
                    chunks.append({
                        "file_path": file_path,
                        "start_line": start_line,
                        "end_line": end_line,
                        "content": "\n".join(block_lines),
                        "name": lines[i].strip()[:60],
                        "type": "block",
                    })
                    i = j
                    break
                j += 1

        i += 1

    if not chunks:
        return fallback_chunking(file_path, code)
    return chunks


# ── Dispatcher ────────────────────────────────────────────────────────────────

# FIX #12 (partial): extended to include JSX/TSX and common config formats.
_BRACE_EXTENSIONS = {".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".h", ".go"}
_FALLBACK_EXTENSIONS = {
    ".css", ".scss", ".html", ".yml", ".yaml", ".toml",
    ".json", ".md", ".txt", ".sh", ".bash",
}


def chunk_file(file_path: str) -> list:
    """Detect file type and chunk accordingly."""
    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
    except Exception:
        return []

    _, ext = os.path.splitext(file_path.lower())

    if ext == ".py":
        return chunk_python_file(file_path, code)
    elif ext in _BRACE_EXTENSIONS:
        return chunk_brace_file(file_path, code)
    elif ext in _FALLBACK_EXTENSIONS:
        return fallback_chunking(file_path, code, chunk_size=40)
    else:
        return []
