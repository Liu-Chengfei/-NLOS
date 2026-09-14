import re
import sys

def main():
    txt = open('docs/zcode_异步手册_v696_20260914_最新-落实报告.md', encoding='utf-8').read()
    lines = txt.splitlines(keepends=False)  # no trailing \n

    # 1. Identify stage headers and their line indices
    stage_headers = []  # list of (line_idx, header_text, stage_num)
    for i, line in enumerate(lines):
        if line.startswith('## Stage'):
            # extract stage number from "[1–30]" etc
            m = re.search(r'\[(\d+)–(\d+)\]', line)
            if m:
                start = int(m.group(1))
                end = int(m.group(2))
                stage_headers.append((i, line, start, end))
        elif line.startswith('## 汇总'):
            stage_headers.append((i, line, None, None))  # summary

    if not stage_headers:
        print("No stage headers found", file=sys.stderr)
        sys.exit(1)

    # 2. Extract clause lines: those matching ^\*\*\[(\d+)\]\*\*
    clause_pattern = re.compile(r'^\*\*\[(\d+)\]\*\*')
    clause_map = {}  # num -> line (full)
    for line in lines:
        m = clause_pattern.match(line)
        if m:
            num = int(m.group(1))
            # keep the first encountered line for this num (we will dedup later)
            if num not in clause_map:
                clause_map[num] = line

    print(f"Found {len(clause_map)} unique clause numbers: {sorted(clause_map.keys())[:10]}...{sorted(clause_map.keys())[-10:]}")

    # 3. Determine preamble: lines before first stage header
    first_stage_idx = stage_headers[0][0]
    preamble = lines[:first_stage_idx]  # includes header lines 1-8 and maybe blank before first stage

    # 4. Build output
    out_lines = []
    out_lines.extend(preamble)

    # Process each stage in order
    for stage_idx, header_text, start, end in stage_headers:
        if start is None:  # summary
            out_lines.append('')  # blank before summary
            out_lines.append(header_text)
            # after this, we will append the remaining lines (from summary start onward) later
            break
        # stage header
        out_lines.append('')  # blank before stage header
        out_lines.append(header_text)
        out_lines.append('')  # blank after header

        # collect clause lines for this stage's range
        stage_clauses = []
        for num in range(start, end+1):
            if num in clause_map:
                stage_clauses.append((num, clause_map[num]))
        # sort by num
        stage_clauses.sort(key=lambda x: x[0])
        for num, line in stage_clauses:
            out_lines.append(line)
            out_lines.append('')  # blank after clause

    # 5. Append summary section (from the summary header line to end)
    # Find where summary starts in original lines
    summary_start_idx = None
    for i, line in enumerate(lines):
        if line.startswith('## 汇总'):
            summary_start_idx = i
            break
    if summary_start_idx is not None:
        # append a blank line before summary if not already
        if out_lines and out_lines[-1] != '':
            out_lines.append('')
        out_lines.extend(lines[summary_start_idx:])

    # Join and write
    new_txt = '\n'.join(out_lines)
    open('docs/zcode_异步手册_v696_20260914_最新-落实报告.md', 'w', encoding='utf-8').write(new_txt)
    print(f"Rewritten file, total lines: {len(out_lines)}")

    # Verify
    verify_txt = new_txt
    clause_nums = [int(m.group(1)) for m in re.finditer(r'\*\*\[(\d+)\]\*\*', verify_txt)]
    print(f"Clause count in new file: {len(clause_nums)}")
    print(f"Range: {min(clause_nums)} to {max(clause_nums)}")
    print(f"All present 1-696: {set(range(1,697)) == set(clause_nums)}")
    # Check order
    ordered = all(clause_nums[i] <= clause_nums[i+1] for i in range(len(clause_nums)-1))
    print(f"Is non-decreasing: {ordered}")

if __name__ == '__main__':
    main()