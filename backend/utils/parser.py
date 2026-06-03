import os
import re

def fuzzy_replace(content: str, search_str: str, replace_str: str) -> str:
    """Whitespace-insensitive replacement of search_str in content."""
    def normalize(s):
        return re.sub(r'\s+', '', s)
        
    norm_search = normalize(search_str)
    if not norm_search:
        return None
        
    orig_chars = []
    norm_indices = []
    
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

def apply_search_replace(repo_dir: str, llm_output: str) -> dict:
    """
    Parses LLM output containing search/replace blocks using a robust state machine.
    Handles unclosed blocks, missing delimiters, and consecutive blocks.
    """
    lines = llm_output.splitlines()
    n = len(lines)
    
    current_file = None
    blocks_by_file = {}
    
    # State can be: 'outside', 'search', 'replace'
    state = 'outside'
    search_lines = []
    replace_lines = []
    
    i = 0
    while i < n:
        line = lines[i]
        line_stripped = line.strip()
        
        # Check for file path markers
        file_match = re.match(r'^(?:FILE|File|filepath|Filepath|Path|path|Target File|Target file)\s*:\s*`?([^`\s]+)`?', line_stripped)
        if not file_match:
            file_match = re.match(r'^###?\s*`?([a-zA-Z0-9_.\-/]+\.[a-zA-Z0-9]+)`?$', line_stripped)
            
        if file_match:
            # Flush existing block if in replace state
            if state == 'replace' and current_file and (search_lines or replace_lines):
                blocks_by_file.setdefault(current_file, []).append({
                    "search": "\n".join(search_lines),
                    "replace": "\n".join(replace_lines)
                })
            current_file = file_match.group(1).strip().lstrip('/\\')
            state = 'outside'
            search_lines = []
            replace_lines = []
            i += 1
            continue
            
        if line_stripped.startswith('<<<<<<< SEARCH'):
            if state == 'replace' and current_file and (search_lines or replace_lines):
                blocks_by_file.setdefault(current_file, []).append({
                    "search": "\n".join(search_lines),
                    "replace": "\n".join(replace_lines)
                })
            state = 'search'
            search_lines = []
            replace_lines = []
            i += 1
            continue
            
        if line_stripped.startswith('======='):
            if state == 'search':
                state = 'replace'
            i += 1
            continue
            
        if line_stripped.startswith('>>>>>>> REPLACE'):
            if state == 'replace' and current_file and (search_lines or replace_lines):
                blocks_by_file.setdefault(current_file, []).append({
                    "search": "\n".join(search_lines),
                    "replace": "\n".join(replace_lines)
                })
            state = 'outside'
            search_lines = []
            replace_lines = []
            i += 1
            continue
            
        # Accumulate code lines
        if state == 'search':
            search_lines.append(line)
        elif state == 'replace':
            replace_lines.append(line)
            
        i += 1
        
    # Flush any unclosed block at the end of parsing
    if state == 'replace' and current_file and (search_lines or replace_lines):
        blocks_by_file.setdefault(current_file, []).append({
            "search": "\n".join(search_lines),
            "replace": "\n".join(replace_lines)
        })
        
    modified_files = []
    errors = []
    
    for file_rel_path, blocks in blocks_by_file.items():
        if file_rel_path == "unknown":
            continue
            
        full_path = os.path.join(repo_dir, file_rel_path)
        if not os.path.exists(full_path):
            is_creation = any(b["search"].strip() == "" for b in blocks)
            if is_creation:
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write("")
            else:
                found = False
                for root, _, files in os.walk(repo_dir):
                    for file in files:
                        if file == os.path.basename(file_rel_path):
                            possible_path = os.path.join(root, file)
                            possible_rel = os.path.relpath(possible_path, repo_dir)
                            if not any(part in possible_rel.split(os.sep) for part in ('.git', 'node_modules', 'venv', '__pycache__', 'dist', 'build')):
                                full_path = possible_path
                                file_rel_path = possible_rel
                                found = True
                                break
                    if found:
                        break
                if not found:
                    errors.append(f"File not found in repository: {file_rel_path}")
                    continue
                
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            new_content = content
            for block in blocks:
                search_str = block["search"]
                replace_str = block["replace"]
                
                # Check for exact match
                if search_str in new_content:
                    new_content = new_content.replace(search_str, replace_str, 1)
                else:
                    # Let's try line-ending normalized match
                    search_norm = search_str.replace('\r\n', '\n')
                    content_norm = new_content.replace('\r\n', '\n')
                    
                    if search_norm in content_norm:
                        replace_norm = replace_str.replace('\r\n', '\n')
                        content_norm = content_norm.replace(search_norm, replace_norm, 1)
                        new_content = content_norm
                    else:
                        search_strip = search_str.strip()
                        if search_strip in new_content:
                            new_content = new_content.replace(search_strip, replace_str, 1)
                        else:
                            search_norm_strip = search_norm.strip()
                            content_norm = new_content.replace('\r\n', '\n')
                            if search_norm_strip in content_norm:
                                new_content = content_norm.replace(search_norm_strip, replace_str.replace('\r\n', '\n'), 1)
                            else:
                                # Fallback to whitespace-insensitive fuzzy match!
                                fuzzy_res = fuzzy_replace(new_content, search_str, replace_str)
                                if fuzzy_res is not None:
                                    new_content = fuzzy_res
                                else:
                                    errors.append(f"Could not locate the search block in {file_rel_path}. Block content:\n{search_str}")
                                
            if new_content != content:
                with open(full_path, 'w', encoding='utf-8', newline='') as f:
                    f.write(new_content)
                modified_files.append(file_rel_path)
                
        except Exception as e:
            errors.append(f"Failed to modify {file_rel_path}: {str(e)}")
            
    # Try resolving unknown blocks
    if "unknown" in blocks_by_file and blocks_by_file["unknown"]:
        unknown_blocks = blocks_by_file["unknown"]
        for root, _, files in os.walk(repo_dir):
            for file in files:
                full_path = os.path.join(root, file)
                file_rel = os.path.relpath(full_path, repo_dir)
                if any(part in file_rel.split(os.sep) for part in ('.git', 'node_modules', 'venv', '__pycache__', 'dist', 'build')):
                    continue
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    new_content = content
                    matched_any = False
                    for block in unknown_blocks:
                        search_str = block["search"]
                        replace_str = block["replace"]
                        search_norm = search_str.replace('\r\n', '\n')
                        content_norm = new_content.replace('\r\n', '\n')
                        
                        if search_str in new_content:
                            new_content = new_content.replace(search_str, replace_str, 1)
                            matched_any = True
                        elif search_norm in content_norm:
                            new_content = content_norm.replace(search_norm, replace_str.replace('\r\n', '\n'), 1)
                            matched_any = True
                            
                    if matched_any and new_content != content:
                        with open(full_path, 'w', encoding='utf-8', newline='') as f:
                            f.write(new_content)
                        if file_rel not in modified_files:
                            modified_files.append(file_rel)
                except Exception:
                    pass
                    
    return {
        "modified_files": modified_files,
        "errors": errors
    }
