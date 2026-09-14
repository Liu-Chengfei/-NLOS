#!/usr/bin/env python3
"""
Extract gate function line numbers programmatically for report injection.
Auto-scans precheck_orchestrator.py and outputs a JSON map.
"""
import re
import json
from pathlib import Path

def extract_gates():
    src = Path("src/liquidloc/common/precheck_orchestrator.py")
    content = src.read_text(encoding="utf-8")
    
    # Match function definitions: def check_<ID>(...):
    pattern = r'^def\s+(check_(?:[A-Za-z][A-Za-z0-9_]+))\s*\('
    matches = list(re.finditer(pattern, content, re.MULTILINE))
    
    gate_lines = {}
    for m in matches:
        func_name = m.group(1)
        # Calculate line number (count newlines before match start)
        line_num = content[:m.start()].count('\n') + 1
        gate_lines[func_name] = line_num
    
    return gate_lines

if __name__ == "__main__":
    gates = extract_gates()
    
    # Output sorted by line number
    sorted_gates = sorted(gates.items(), key=lambda x: x[1])
    print(json.dumps(sorted_gates, indent=2))
    
    # Also write to file for easy copy-paste
    out = Path("scripts/.gate_line_numbers.json")
    out.write_text(json.dumps(sorted_gates, indent=2), encoding="utf-8")
    print(f"\n→ Saved to {out}", flush=True)
