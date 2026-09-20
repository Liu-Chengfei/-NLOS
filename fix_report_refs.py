import re

with open('docs/最终交付_异步手册_424条_20260917-第0阶段落实报告.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: checklist_gate.md:1-300 → :1-276
content = content.replace('rules/checklist_gate.md:1-300', 'rules/checklist_gate.md:1-276')

# Fix 2: 治理载体分散布局.md:1-23 → :1-22
content = content.replace('rules/治理载体分散布局.md:1-23', 'rules/治理载体分散布局.md:1-22')

# Fix 3: 状态只认磁盘.md:29-50 → :29-32
content = content.replace('rules/状态只认磁盘.md:29-50', 'rules/状态只认磁盘.md:29-32')

# Fix 4: rules/落实报告格式.md → rules/落实报告格式规范.md
content = content.replace('rules/落实报告格式.md', 'rules/落实报告格式规范.md')

with open('docs/最终交付_异步手册_424条_20260917-第0阶段落实报告.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("修复完成")

# Verify
with open('docs/最终交付_异步手册_424条_20260917-第0阶段落实报告.md', 'r', encoding='utf-8') as f:
    content = f.read()

issues = []
for path, wrong in [('rules/checklist_gate.md:1-300', ':1-300'),
                     ('rules/治理载体分散布局.md:1-23', ':1-23'),
                     ('rules/状态只认磁盘.md:29-50', ':29-50'),
                     ('rules/落实报告格式.md', '落实报告格式.md')]:
    if wrong in content:
        issues.append(f'❌ 仍存在: {wrong}')
    else:
        print(f'✅ 已清除: {wrong}')

if not issues:
    print('\n✅ 所有问题已修复')
else:
    for i in issues:
        print(i)
