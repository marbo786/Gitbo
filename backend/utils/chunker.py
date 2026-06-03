import os
import ast
import re

def fallback_chunking(file_path: str, code: str, chunk_size: int = 50) -> list:
    """Sliding window chunking for non-structured files or files where parsing failed."""
    chunks = []
    lines = code.splitlines()
    if not lines:
        return []
    
    i = 0
    while i < len(lines):
        end = min(i + chunk_size, len(lines))
        content = "\n".join(lines[i:end])
        chunks.append({
            "file_path": file_path,
            "start_line": i + 1,
            "end_line": end,
            "content": content,
            "name": f"lines {i+1}-{end}",
            "type": "raw_lines"
        })
        i += max(1, chunk_size - 10) # 10 lines overlap
    return chunks

def chunk_python_file(file_path: str, code: str) -> list:
    """Parse Python code using built-in AST to split by functions, classes, and methods."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return fallback_chunking(file_path, code)
        
    chunks = []
    lines = code.splitlines()
    num_lines = len(lines)
    
    class CodeVisitor(ast.NodeVisitor):
        def visit_ClassDef(self, node):
            start = node.lineno
            end = getattr(node, 'end_lineno', num_lines)
            content = "\n".join(lines[start-1:end])
            chunks.append({
                "file_path": file_path,
                "start_line": start,
                "end_line": end,
                "content": content,
                "name": f"class {node.name}",
                "type": "class"
            })
            self.generic_visit(node)
            
        def visit_FunctionDef(self, node):
            start = node.lineno
            end = getattr(node, 'end_lineno', num_lines)
            content = "\n".join(lines[start-1:end])
            chunks.append({
                "file_path": file_path,
                "start_line": start,
                "end_line": end,
                "content": content,
                "name": f"def {node.name}",
                "type": "function"
            })
            self.generic_visit(node)
            
        def visit_AsyncFunctionDef(self, node):
            start = node.lineno
            end = getattr(node, 'end_lineno', num_lines)
            content = "\n".join(lines[start-1:end])
            chunks.append({
                "file_path": file_path,
                "start_line": start,
                "end_line": end,
                "content": content,
                "name": f"async def {node.name}",
                "type": "function"
            })
            self.generic_visit(node)

    visitor = CodeVisitor()
    visitor.visit(tree)
    
    if not chunks:
        return fallback_chunking(file_path, code)
    return chunks

def chunk_brace_file(file_path: str, code: str) -> list:
    """Parse brace-delimited languages (JS, TS, Go, Java, C++) to split by function/class blocks."""
    chunks = []
    lines = code.splitlines()
    n = len(lines)
    
    # Common patterns for declarations
    patterns = [
        r'\bclass\s+\w+',
        r'\bfunction\s+\w*',
        r'\bfunc\s+',
        r'\b\w+\s*\([^)]*\)\s*\{',  # generic function/method signature
        r'\bconst\s+\w+\s*=\s*(?:\([^)]*\)|\w+)\s*=>'  # arrow function
    ]
    combined_pattern = re.compile('|'.join(patterns))
    
    i = 0
    while i < n:
        line = lines[i]
        if combined_pattern.search(line):
            start_line = i + 1
            found_open = False
            brace_count = 0
            block_lines = []
            
            j = i
            while j < n:
                block_lines.append(lines[j])
                current_line = lines[j]
                
                for char in current_line:
                    if char == '{':
                        found_open = True
                        brace_count += 1
                    elif char == '}':
                        if found_open:
                            brace_count -= 1
                            if brace_count == 0:
                                end_line = j + 1
                                content = "\n".join(block_lines)
                                chunks.append({
                                    "file_path": file_path,
                                    "start_line": start_line,
                                    "end_line": end_line,
                                    "content": content,
                                    "name": line.strip()[:60],
                                    "type": "block"
                                })
                                i = j  # Skip past the end of this block
                                break
                if found_open and brace_count == 0:
                    break
                j += 1
        i += 1
        
    if not chunks:
        return fallback_chunking(file_path, code)
    return chunks

def chunk_file(file_path: str) -> list:
    """Detect file type and chunk it using the corresponding chunker."""
    if not os.path.exists(file_path):
        return []
        
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
    except Exception:
        return []
        
    _, ext = os.path.splitext(file_path.lower())
    
    if ext == '.py':
        return chunk_python_file(file_path, code)
    elif ext in ('.js', '.ts', '.java', '.cpp', '.h', '.go'):
        return chunk_brace_file(file_path, code)
    else:
        return fallback_chunking(file_path, code)
