import os
import re
from pathlib import Path


# ── Path safety ───────────────────────────────────────────────────────────────

def _safe_resolve(repo_dir: str, rel_path: str) -> str | None:
    """
    FIX #2: Resolve rel_path relative to repo_dir and verify the result stays
    inside repo_dir.  Returns the absolute path if safe, or None if the path
    escapes the repository root (e.g. FILE: ../../etc/passwd).
    """
    repo_root = Path(repo_dir).resolve()
    target = (repo_root / rel_path).resolve()
    try:
        target.relative_to(repo_root)  # raises ValueError if outside root
        return str(target)
    except ValueError:
        return None


# ── Fuzzy replacement ─────────────────────────────────────────────────────────

# Minimum non-whitespace characters the search block must contain before we
# allow fuzzy (whitespace-insensitive) matching.  Short blocks are too
# ambiguous and can corrupt unrelated code.
_FUZZY_MIN_CHARS = 20


def fuzzy_replace(content: str, search_str: str, replace_str: str) -> str | None:
    """
    Whitespace-insensitive replacement of search_str in content.

    FIX #14: We now reject fuzzy matching when the normalised search string is
    shorter than _FUZZY_MIN_CHARS, because very short patterns match too many
    unrelated locations.
    """
    def normalize(s: str) -> str:
        return re.sub(r"\s+", "", s)

    norm_search = normalize(search_str)
    if not norm_search or len(norm_search) < _FUZZY_MIN_CHARS:
        return None

    orig_chars: list[str] = []
    norm_indices: list[int] = []

    for idx, char in enumerate(content):
        if not char.isspace():
            orig_chars.append(char)
            norm_indices.append(idx)

    norm_content = "".join(orig_chars)
    match_idx = norm_content.find(norm_search)
    if match_idx == -1:
        return None

    start_orig = norm_indices[match_idx]
    end_orig = norm_indices[match_idx + len(norm_search) - 1] + 1
    return content[:start_orig] + replace_str + content[end_orig:]


# ── Search/Replace parser ─────────────────────────────────────────────────────

def apply_search_replace(repo_dir: str, llm_output: str) -> dict:
    """
    Parse LLM output containing search/replace blocks and apply them to the
    cloned repository.

    State machine states: 'outside' | 'search' | 'replace'
    """
    lines = llm_output.splitlines()
    n = len(lines)

    current_file: str | None = None
    blocks_by_file: dict[str, list] = {}

    state = "outside"
    search_lines: list[str] = []
    replace_lines: list[str] = []

    i = 0
    while i < n:
        line = lines[i]
        line_stripped = line.strip()

        # ── File path markers ──────────────────────────────────────────────
        file_match = re.match(
            r"^(?:FILE|File|filepath|Filepath|Path|path|Target File|Target file)\s*:\s*`?([^`\s]+)`?",
            line_stripped,
        )
        if not file_match:
            file_match = re.match(
                r"^###?\s*`?([a-zA-Z0-9_.\-/]+\.[a-zA-Z0-9]+)`?$",
                line_stripped,
            )

        if file_match:
            if state == "replace" and current_file and (search_lines or replace_lines):
                blocks_by_file.setdefault(current_file, []).append(
                    {"search": "\n".join(search_lines), "replace": "\n".join(replace_lines)}
                )
            raw_path = file_match.group(1).strip().lstrip("/\\")
            current_file = raw_path
            state = "outside"
            search_lines = []
            replace_lines = []
            i += 1
            continue

        if line_stripped.startswith("<<<<<<< SEARCH"):
            if current_file is None:
                current_file = "unknown"
            if state == "replace" and current_file and (search_lines or replace_lines):
                blocks_by_file.setdefault(current_file, []).append(
                    {"search": "\n".join(search_lines), "replace": "\n".join(replace_lines)}
                )
            state = "search"
            search_lines = []
            replace_lines = []
            i += 1
            continue

        if line_stripped.startswith("======="):
            if state == "search":
                state = "replace"
            i += 1
            continue

        if line_stripped.startswith(">>>>>>> REPLACE"):
            if state == "replace" and current_file and (search_lines or replace_lines):
                blocks_by_file.setdefault(current_file, []).append(
                    {"search": "\n".join(search_lines), "replace": "\n".join(replace_lines)}
                )
            state = "outside"
            search_lines = []
            replace_lines = []
            i += 1
            continue

        if state == "search":
            search_lines.append(line)
        elif state == "replace":
            replace_lines.append(line)

        i += 1

    # Flush any unclosed block
    if state == "replace" and current_file and (search_lines or replace_lines):
        blocks_by_file.setdefault(current_file, []).append(
            {"search": "\n".join(search_lines), "replace": "\n".join(replace_lines)}
        )

    modified_files: list[str] = []
    errors: list[str] = []

    for file_rel_path, blocks in blocks_by_file.items():
        # ── FIX #15: Reject unknown blocks instead of walking the whole repo ──
        if file_rel_path == "unknown":
            errors.append(
                "Parser encountered blocks without a FILE: header (unknown). "
                "Skipped to avoid modifying wrong files."
            )
            continue

        # ── FIX #2: Containment check — reject path-traversal attempts ────
        safe_full_path = _safe_resolve(repo_dir, file_rel_path)
        if safe_full_path is None:
            errors.append(
                f"Rejected unsafe file path (possible path traversal): {file_rel_path}"
            )
            continue

        full_path = safe_full_path

        if not os.path.exists(full_path):
            is_creation = any(b["search"].strip() == "" for b in blocks)
            if is_creation:
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write("")
            else:
                # FIX #13: Only use basename fallback when there is exactly
                # ONE unambiguous match; warn and skip if multiple exist.
                basename = os.path.basename(file_rel_path)
                matches = []
                for root, _, files in os.walk(repo_dir):
                    for fname in files:
                        if fname == basename:
                            candidate = os.path.join(root, fname)
                            rel = os.path.relpath(candidate, repo_dir)
                            excluded = (".git", "node_modules", "venv", "__pycache__", "dist", "build")
                            if not any(part in rel.split(os.sep) for part in excluded):
                                matches.append(candidate)

                if len(matches) == 1:
                    full_path = matches[0]
                    file_rel_path = os.path.relpath(full_path, repo_dir)
                elif len(matches) > 1:
                    errors.append(
                        f"Ambiguous file path '{file_rel_path}' — {len(matches)} files share "
                        f"that basename. Skipped to avoid modifying the wrong file."
                    )
                    continue
                else:
                    errors.append(f"File not found in repository: {file_rel_path}")
                    continue

        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            new_content = content
            for block in blocks:
                search_str = block["search"]
                replace_str = block["replace"]

                # 1. Exact match
                if search_str in new_content:
                    new_content = new_content.replace(search_str, replace_str, 1)
                    continue

                # 2. Line-ending-normalised match
                search_lf = search_str.replace("\r\n", "\n")
                content_lf = new_content.replace("\r\n", "\n")
                if search_lf in content_lf:
                    new_content = content_lf.replace(
                        search_lf, replace_str.replace("\r\n", "\n"), 1
                    )
                    continue

                # 3. Strip-normalised match
                search_strip = search_str.strip()
                if search_strip and search_strip in new_content:
                    new_content = new_content.replace(search_strip, replace_str, 1)
                    continue

                search_lf_strip = search_lf.strip()
                if search_lf_strip and search_lf_strip in content_lf:
                    new_content = content_lf.replace(
                        search_lf_strip, replace_str.replace("\r\n", "\n"), 1
                    )
                    continue

                # 4. FIX #14: Fuzzy match — only for sufficiently long blocks
                fuzzy_res = fuzzy_replace(new_content, search_str, replace_str)
                if fuzzy_res is not None:
                    new_content = fuzzy_res
                else:
                    errors.append(
                        f"Could not locate the search block in {file_rel_path}.\n"
                        f"Block (first 200 chars): {search_str[:200]}"
                    )

            if new_content != content:
                with open(full_path, "w", encoding="utf-8", newline="") as f:
                    f.write(new_content)
                modified_files.append(file_rel_path)

        except Exception as e:
            errors.append(f"Failed to modify {file_rel_path}: {e}")

    return {"modified_files": modified_files, "errors": errors}
