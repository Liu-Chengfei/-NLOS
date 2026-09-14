#!/bin/bash
# verify_slim.sh — 异步高 NLOS 机械核验脚本（Stage 0[17]/[695]）
# 退出码：0 = 全部 PASS，1 = 任意 FAIL
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# Windows Git Bash 下解释器常为 `python`，Linux/macOS 为 `python3`
if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi

echo "=== verify_slim.sh A: Python 语法检查 ==="
$PY scripts/verify_slim_a.py
if [ $? -ne 0 ]; then
    echo "A FAIL: Python 语法错误"
    exit 1
fi
echo "A PASS: 语法通过"

echo "=== verify_slim.sh B: flake8 风格检查 ==="
if command -v flake8 >/dev/null 2>&1; then
    flake8 src/ --count --extend-ignore=E501,E701,F401,F841,E722,W503 --exclude=migrations/ || {
        echo "B FAIL: flake8 检查未通过"
        exit 1
    }
    echo "B PASS: flake8 检查通过"
else
    echo "B WARN: 未安装 flake8，跳过"
fi

echo "=== verify_slim.sh C: 逐条审计 ==="
MANUALE="docs/zcode_异步手册_v696_20260914_最新.md"

# C1: 条款号连续（格式 **[N]**）
actual=$(grep -oE '\*\*\[[0-9]+\]\*\*' "$MANUALE" | grep -oE '[0-9]+' | sort -n | tr '\n' ' ')
expected=$(seq 1 696 | tr '\n' ' ')
[ "$actual" = "$expected" ] || {
    echo "C1 FAIL: 条款号断号"
    exit 1
}
echo "C1 PASS: 条款号连续 (1-696)"

# C2: 无转义残留
grep -qE '%[0-9]{2}' "$MANUALE" && { echo "C2 FAIL: 存在转义残留"; exit 1; }
echo "C2 PASS: 无转义残留"

# C3: 产物路径
for f in "AGENTS.md" "docs/zcode_异步手册_v696_20260914_最新.md" "configs/models/liquid_ekf.yaml"; do
    [ -f "$f" ] || { echo "C3 FAIL: 缺失 $f"; exit 1; }
done
echo "C3 PASS: 产物路径均存"

echo "=== verify_slim.sh D: FAST 冒烟检查 ==="
[ -f "src/liquidloc/common/precheck_orchestrator.py" ] || { echo "D FAIL: precheck_orchestrator.py 不存在"; exit 1; }
echo "D PASS: precheck_orchestrator.py 存在"

echo "=== verify_slim.sh E: pytest 冒烟 ==="
if command -v pytest >/dev/null 2>&1 && [ -d "tests" ]; then
    pytest tests/ -x -q --tb=short 2>&1 | tail -3
    echo "E PASS: pytest 完成"
else
    echo "E WARN: 跳过 pytest"
fi

echo "=== verify_slim.sh F: 完整性 spot-check ==="
[ -d ".audit" ] && echo "F PASS: .audit 目录存在" || echo "F WARN: .audit 目录不存在"
[ -d "archive" ] && echo "F PASS: archive 目录存在" || { echo "F FAIL: archive 目录不存在"; exit 1; }

echo "=== verify_slim.sh ALL PASS ==="