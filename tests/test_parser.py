import pytest
from backend.utils.parser import fuzzy_replace, apply_search_replace

def test_fuzzy_replace_short_string_rejected():
    # Less than _FUZZY_MIN_CHARS
    content = "def add(a, b):\n    return a + b\n"
    search_str = "return a + b"
    replace_str = "return a + b + 0"
    assert fuzzy_replace(content, search_str, replace_str) is None

def test_fuzzy_replace_success():
    content = "def calculate_total(items):\n    sum = 0\n    for item in items:\n        sum += item.price\n    return sum\n"
    search_str = "    sum = 0\n    for item in items:\n        sum += item.price\n"
    replace_str = "    sum = sum(item.price for item in items)\n"
    
    # search_str has > 20 non-whitespace chars, should work
    res = fuzzy_replace(content, search_str, replace_str)
    assert res is not None
    assert "sum(item.price for item in items)" in res

def test_apply_search_replace_path_traversal(tmp_path):
    repo_dir = str(tmp_path)
    
    llm_output = """
FILE: ../../../etc/passwd
<<<<<<< SEARCH
root:x:0:0:root:/root:/bin/bash
=======
root:x:0:0:root:/root:/bin/zsh
>>>>>>> REPLACE
"""
    result = apply_search_replace(repo_dir, llm_output)
    assert len(result["modified_files"]) == 0
    assert any("possible path traversal" in err for err in result["errors"])

def test_apply_search_replace_unknown_block(tmp_path):
    repo_dir = str(tmp_path)
    
    llm_output = """
<<<<<<< SEARCH
foo
=======
bar
>>>>>>> REPLACE
"""
    result = apply_search_replace(repo_dir, llm_output)
    assert len(result["modified_files"]) == 0
    assert any("unknown" in err.lower() for err in result["errors"])

def test_apply_search_replace_success(tmp_path):
    repo_dir = tmp_path
    file_path = repo_dir / "main.py"
    file_path.write_text("print('hello world')\n", encoding="utf-8")
    
    llm_output = """
FILE: main.py
<<<<<<< SEARCH
print('hello world')
=======
print('hello universe')
>>>>>>> REPLACE
"""
    result = apply_search_replace(str(repo_dir), llm_output)
    assert len(result["modified_files"]) == 1
    assert result["modified_files"][0] == "main.py"
    assert file_path.read_text(encoding="utf-8") == "print('hello universe')\n"
