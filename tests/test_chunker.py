import pytest
from backend.utils.chunker import _count_braces_in_line, chunk_python_file

def test_count_braces_in_line():
    assert _count_braces_in_line("function test() {") == (1, 0)
    assert _count_braces_in_line("}") == (0, 1)
    
    # Strings should be ignored
    assert _count_braces_in_line('const a = "{";') == (0, 0)
    assert _count_braces_in_line('console.log("}");') == (0, 0)
    
    # Comments should be ignored
    assert _count_braces_in_line('// {') == (0, 0)
    assert _count_braces_in_line('// }') == (0, 0)

def test_chunk_python_file():
    code = """import os
import sys

# module header
MAX_VAL = 100

class MyClass:
    def method1(self):
        pass

def top_level_func():
    pass
"""
    chunks = chunk_python_file("test.py", code)
    assert len(chunks) == 3
    
    types = [c["type"] for c in chunks]
    assert "module_header" in types
    assert "class" in types
    assert "function" in types
    
    # Ensure method1 is NOT its own chunk to prevent duplication (fix #17)
    names = [c["name"] for c in chunks]
    assert "def method1" not in names
