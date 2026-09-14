#!/usr/bin/env python3
"""verify_slim_a.py — Python syntax checker helper"""
import py_compile
import pathlib
import sys

errors = []
for path in pathlib.Path("src").rglob("*.py"):
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as e:
        errors.append(str(e))

if errors:
    print(f"FAIL: {len(errors)} files with syntax errors")
    for e in errors[:10]:
        print(f"  {e}")
    sys.exit(1)
print(f"PASS: all .py files syntax OK")
sys.exit(0)
