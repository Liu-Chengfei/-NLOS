#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成AGENTS.md智能体指南脚本"""
import sys
from pathlib import Path

def main():
    project_root = Path.cwd()
    agents_path = project_root / "AGENTS.md"
    manual_path = project_root / "docs" / "最终交付_异步手册_424条_20260917.md"
    
    if manual_path.exists():
        content = manual_path.read_text(encoding="utf-8")
        agents_path.write_text(content, encoding="utf-8")
        return 0
    return 1

if __name__ == "__main__":
    sys.exit(main())
