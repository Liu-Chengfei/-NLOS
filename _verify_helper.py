#!/usr/bin/env python3
"""Verify manual clause markers and key phrases."""
import re, sys

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    content = f.read()

# Count clause markers: **\[N] where \ is literal backslash in source
# The raw bytes are: * * \ [ N ]
# In Python regex: r'\*\*\\[(\d+)'
# - \* = literal *
# - \\ = literal backslash (in raw string, \\ is two backslashes = one literal backslash in regex)
# - \[ = literal [
# - (\d+) = capture group for digits

# Let's just search for the literal pattern
marker_prefix = '**\\['  # ** + backslash + [
count = 0
nums = set()
idx = 0
while True:
    pos = content.find(marker_prefix, idx)
    if pos < 0:
        break
    count += 1
    # Extract the number after [
    rest = content[pos + len(marker_prefix):]
    m = re.match(r'(\d+)', rest)
    if m:
        nums.add(int(m.group(1)))
    idx = pos + len(marker_prefix)

print(f"clause_marker_count={count}")
print(f"unique_clause_numbers={len(nums)}")
if nums:
    print(f"clause_range={min(nums)}-{max(nums)}")

# Find Stage 7 first clause
lines = content.split('\n')
in_s7 = False
stage7_first = None
for line in lines:
    if line.startswith('## Stage 7'):
        in_s7 = True
        continue
    if in_s7:
        if marker_prefix in line:
            m = re.search(rf'{re.escape(marker_prefix)}(\d+)', line)
            if m:
                stage7_first = m.group(1)
                break

print(f"stage7_first_clause={stage7_first or 'none'}")
