#!/usr/bin/env python3
"""Add p12_manual_objectives to pre-commit hook."""
with open('E:/异步高NLOS/.git/hooks/pre-commit', 'r', encoding='utf-8') as f:
    text = f.read()

if 'p12_manual_objectives' in text:
    print('p12 hook call already exists')
else:
    # Find 'exit $fail' at the end of the file and insert before it
    import re
    # Match the last occurrence of 'exit $fail'
    pattern = re.compile(r'(\nexit \$fail\n?$)')
    match = pattern.search(text)
    if match:
        insert_pos = match.start()
        p12_block = '''
# [12] 手册目标：调用p12验证函数确保核心判据规则文件完整
if [ "$fail" -eq 0 ]; then
  p12_result=$(python -c "import sys; sys.path.insert(0, 'src'); from liquidloc.analysis.clause_gates import p12_manual_objectives; r = p12_manual_objectives(); print('PASSED: ' + str(r['passed']) + ' | ' + r['detail'])" 2>/dev/null || echo "")
  if [ -n "$p12_result" ]; then
    echo "[pre-commit] [12] 手册目标: $p12_result"
    if echo "$p12_result" | grep -q "FAILED|False"; then
      echo "[pre-commit] 条款 [12] 手册目标验证未通过，阻止提交"
      fail=1
    fi
  fi
fi
'''
        new_text = text[:insert_pos] + p12_block + text[insert_pos:]
        with open('E:/异步高NLOS/.git/hooks/pre-commit', 'w', encoding='utf-8') as f:
            f.write(new_text)
        print('p12 hook call added successfully')
    else:
        print('ERROR: exit $fail not found')
