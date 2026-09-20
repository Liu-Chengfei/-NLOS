#!/usr/bin/env bash
# verify_slim.sh —— 执行手册·落实报告核验器（与手册"落实报告格式·八字段逐字段规范"逐条对齐）
# 用法: bash verify_slim.sh <手册.md> [auto|phase0|phase1]   （默认 auto: 有主落实报告=phase1,否则=phase0）
# 核验: A 手册结构(7阶段制+标题纪律) | B 落实报告覆盖(含第0阶段治理层) | C 报告格式(八字段逐字段) | D 代码落点 | E 清场
# 阶段感知: phase0=第0阶段(治理落实期): 主落实报告/Stage1-6代码落点属第1阶段,缺失为预期不计FAIL;
#           phase1=第1阶段(代码落实期): 主落实报告必须存在,缺失即FAIL。auto 自动按文件存在性判定。
set -u
MANUAL="${1:?用法: bash verify_slim.sh <手册.md> [auto|phase0|phase1]}"
[ -f "$MANUAL" ] || { echo "手册不存在: $MANUAL"; exit 2; }
PHASE_ARG="${2:-auto}"
case "$PHASE_ARG" in auto|phase0|phase1) ;; *) echo "未知阶段: $PHASE_ARG (须为 auto/phase0/phase1)"; exit 2;; esac
PASS=0; FAIL=0
chk(){ if [ "$2" = "1" ]; then echo "✓ $1"; PASS=$((PASS+1)); else echo "✗ $1"; FAIL=$((FAIL+1)); fi; }

echo "========== A. 手册结构核验 =========="
# A1 总纲关键要素(三阶段制/0清场/落实报告格式/逐条落实工序/阶段执行地图/手册目标/底线/语言铁律/报告顺序铁律/第0阶段/归属阶段)
# 关键词映射：手册实际措辞 → verify期望(多词/变体兼容)
for kw_pair in "三阶段制|第0阶段|第1阶段|第2阶段|治理层" "0 清场" "落实报告格式" "逐条落实工序" "阶段执行地图" "手册目标" "底线" "语言铁律" "报告顺序铁律|不跳号|不重复|不倒序" "第 0 阶段" "归属阶段|条款归属|治理层.*Stage"; do
  kw="${kw_pair%%|*}"  # 主关键词(用于显示)
  patterns="${kw_pair#*|}"  # 实际匹配模式(用第一个|分隔后的所有词做或匹配)
  # 取patterns中第一个词作为grep模式(简化处理:用|分隔的词组)
  grep_pat=$(echo "$patterns" | sed 's/|/\\|/g')
  c=$(grep -cE "$grep_pat" "$MANUAL" || true); chk "A1 总纲含[$kw]($c处)" "$([ "$c" -ge 1 ] && echo 1 || echo 0)"
  c=$(grep -c "$kw" "$MANUAL" || true); chk "A1 总纲含[$kw]($c处)" "$([ "$c" -ge 1 ] && echo 1 || echo 0)"
done
# A2 Stage 完整性(1-7 逐号存在,防缺号/重复)
STAGE_MISS=""
for s in 1 2 3 4 5 6 7; do
  if ! grep -qE "^## Stage $s ·" "$MANUAL"; then STAGE_MISS="$STAGE_MISS $s"; fi
done
c=$(grep -cE '^## Stage [0-9]' "$MANUAL" || true)
if [ -z "$STAGE_MISS" ] && [ "$c" -eq 7 ]; then a2ok=1; else a2ok=0; fi
chk "A2 Stage 章节(共$c个;缺号:$STAGE_MISS;须=7且1-7逐号存在)" "$a2ok"
# A2b Stage 0 总纲章节(7阶段制要求总纲独立为 ## 0 · 执行总纲)
c0=$(grep -cE '^## 0 · 执行总纲' "$MANUAL" || true); chk "A2b Stage 0 总纲章节($c0个)" "$([ "$c0" -ge 1 ] && echo 1 || echo 0)"
# A2c' 治理层首条[1]存在(Stage 0 之前有治理层条款[1],非游离说明文字;第0阶段必须逐字抄录落实)
# 手册使用 **\[N]\*\* 格式: 用 \*\*\\[[0-9]+\\]\*\* 匹配(在 regex 中 \\ = literal backslash, \[ = literal [)
gov1=$(grep -cE '\*\*\\[[0-9]+\\]\*\*' "$MANUAL" || true)
chk "A2c' 治理层首条[1]存在($gov1处;Stage0前有治理层条款[1],须逐字抄录落实)" "$([ "$gov1" -ge 1 ] && echo 1 || echo 0)"
# A2c 标题纪律:Stage 0(总纲)之后正文只允许 ## Stage N 标题;## 级总数=8(总纲+7Stage),任何其它 ## 小标题即 FAIL
ch=$(grep -cE '^## ' "$MANUAL" || true)
chk "A2c 标题纪律(## 级共$ch个,须=8:仅总纲+7个Stage;其它##小标题即FAIL)" "$([ "$ch" -eq 8 ] && echo 1 || echo 0)"
# A3 条款数量(去重,全部条款:治理+落实+执行)
# 手册使用 **\[N]\*\* 格式(反斜杠转义括号/星号防markdown链接解析)
total=$(grep -cE '\*\*\\[[0-9]+\\]\*\*' "$MANUAL" || true)
chk "A3 代码条款总数($total条)" "$([ "$total" -ge 300 ] && echo 1 || echo 0)"
# A3b 落实类条款上限 = Stage 7 首条款号 - 1(Stage 7 为第2阶段执行/决策类,不写代码落实报告;落实报告只覆盖治理[1]-[N]+Stage1-6)
stage7_first=$(awk '/^## Stage 7 ·/{f=1} f && match($0, /\*\*\\[[0-9]+\\\]\*\*/){s=substr($0, RSTART+4, RLENGTH-8); gsub(/[^0-9]/,"",s); print s; exit}' "$MANUAL")
if [ -n "$stage7_first" ] && [ "$stage7_first" -gt 1 ]; then
  impl_total=$((stage7_first - 1))
else
  impl_total="$total"
fi
echo "   （Stage 7 首条款号=$stage7_first → 落实报告覆盖条款上限=$impl_total）"
echo ""


echo "========== B. 落实报告覆盖核验 =========="
REP="${MANUAL%.md}-落实报告.md"
REP0="${MANUAL%.md}-第0阶段落实报告.md"
# 阶段感知: phase0 时主落实报告缺失=预期(第1阶段产物); phase1/auto(有主报告) 时缺失=FAIL
if [ -f "$REP" ]; then PHASE="phase1"; else PHASE="phase0"; fi
[ "$PHASE_ARG" != "auto" ] && PHASE="$PHASE_ARG"
echo "  （当前核验阶段: $PHASE）"
# B0 第0阶段落实报告:必须存在、逐条 ✅、条款号引用文件:行号（治理层条款号 = 手册总纲治理条款实际编号，如 [1]-[6]/[1]-[4]，不带 0- 前缀）
# 报告中使用 [N] 格式(无反斜杠) 或 **[N]** 格式
if [ -f "$REP0" ]; then
  rep0_n=$(grep -cE '\[([0-9]+)\]' "$REP0" || true)
  rep0_bad=$(grep -cE '\[[0-9]+\].*❌' "$REP0" || true)
  rep0_loc=$(grep -cE '\[[0-9]+\].*[:：]\s*[0-9]+|\[[0-9]+\].*第\s*[0-9]+\s*[行条]' "$REP0" || true)
  chk "B0 第0阶段报告(${rep0_n}条,未落实${rep0_bad},引用文件:行号${rep0_loc}条)" "$([ "$rep0_bad" = "0" ] && [ "$rep0_loc" -ge 1 ] && echo 1 || echo 0)"
  # B0a 总纲条款[1]已落实:第0阶段报告首条=治理层[1]执行总纲固化,且落实载体体现总纲七要素(三阶段制/0清场/落实报告格式/归属总则/阶段执行地图/手册目标/底线),逐字抄录字段非空
  g1_line=$(grep -E '\[1\]' "$REP0" | head -1 || true)
  if [ -n "$g1_line" ]; then
    g1_term=$(echo "$g1_line" | sed 's/.*条款: //;s/| 代码:.*//' | xargs)
    g1_gov=0
    for gkw in 三阶段制 0\ 清场 落实报告格式 归属 阶段执行地图 手册目标 底线; do
      if echo "$g1_line" | grep -q "$gkw"; then g1_gov=$((g1_gov+1)); fi
    done
    g1_ok=0
    if [ -n "$g1_term" ] && ! echo "$g1_term" | grep -qE '^见手册|^见第|^同上' && [ "$g1_gov" -ge 3 ]; then g1_ok=1; fi
    chk "B0a 总纲条款[1]已逐字抄录落实(抄录非空/覆盖治理要素${g1_gov}类须≥3;须先写第0阶段并覆盖总纲全部内容)" "$g1_ok"
  else
    chk "B0a 总纲条款[1]已逐字抄录落实(缺,第0阶段报告首条必须是治理层[1]执行总纲固化)" 0
  fi
else
  chk "B0 第0阶段落实报告存在(缺,须在第0阶段生成并逐条引用文件:行号)" 0
fi
if [ -f "$REP" ]; then
  rep_n=$(grep -cE '\[[0-9]+\]' "$REP" || true)
  chk "B1 报告行数($rep_n/$impl_total；须≥落实类条款上限,Stage7执行类不写报告)" "$([ "$rep_n" -ge "$impl_total" ] && echo 1 || echo 0)"
  bad=$(grep -cE '\[[0-9]+\].*❌' "$REP" || true)
  chk "B2 未落实条款($bad条)" "$([ "$bad" = "0" ] && echo 1 || echo 0)"
  # B3 报告顺序:条款号必须严格递增(禁止跳号/倒序/合并)
  seq_bad=0; seq_prev=0; seq_first=1
  while IFS= read -r seqline; do
    n=$(echo "$seqline" | sed -E 's/.*\[([0-9]+)\].*/\1/')
    case "$n" in (*[!0-9]*|'') continue;; esac
    if [ "$seq_first" = "1" ]; then seq_prev=$n; seq_first=0; continue; fi
    if [ "$n" -le "$seq_prev" ]; then seq_bad=$((seq_bad+1)); fi
    seq_prev=$n
  done < <(grep -E '^\s*\*?\[[0-9]+\]' "$REP" || true)
  chk "B3 报告顺序(条款号严格递增,乱序$seq_bad处)" "$([ "$seq_bad" = "0" ] && echo 1 || echo 0)"
else
  if [ "$PHASE" = "phase1" ]; then
    chk "B1 报告行数(缺,第1阶段必须生成覆盖$impl_total条的落实报告)" 0
  else
    echo "  ⚠ [phase0] 主落实报告 $REP 未生成 —— 属第1阶段产物,阶段感知跳过 B1-B3/C/D"
  fi
fi
echo ""


echo "========== C. 报告格式核验（八字段逐字段，缺任一字段或字段空话 = 未落实） =========="
if [ -f "$REP" ]; then
  R_FULL=0; R_MISS=0; R_THIN=0; R_TERMMISS=0; R_CODELINEMISS=0; R_REVTHIN=0; R_REVTRACE=0; R_VERTHIN=0; R_EVIMISS=0; R_VERRUN_OK=0; R_VERRUN_BAD=0; R_VERMANUAL=0; R_EVIFILE_OK=0; R_EVIFILE_BAD=0
  while IFS= read -r ln; do
    case "$ln" in
      \[*\]*|\[*)
        # 字段1 条款号 | 字段2 状态 | 字段3 条款(逐字抄录) | 字段4 代码 | 字段5 落实 | 字段6 审查 | 字段7 验证 | 字段8 证据
        has_num=$(echo "$ln" | grep -cE '\[[0-9]+\]' || true)
        has_state=$(echo "$ln" | grep -cE '✅|❌' || true)
        has_term=$(echo "$ln" | grep -c '条款:' || true)
        has_code=$(echo "$ln" | grep -c '代码:' || true)
        has_impl=$(echo "$ln" | grep -c '落实:' || true)
        has_rev=$(echo "$ln" | grep -c '审查:' || true)
        has_ver=$(echo "$ln" | grep -c '验证:' || true)
        has_evi=$(echo "$ln" | grep -c '证据:' || true)
        code=$(echo "$ln" | sed 's/.*代码: //;s/| 落实:.*//')
        if [ "$has_num" -ge 1 ] && [ "$has_state" -ge 1 ] && [ "$has_term" -ge 1 ] && [ "$has_code" -ge 1 ] && [ "$has_impl" -ge 1 ] && [ "$has_rev" -ge 1 ] && [ "$has_ver" -ge 1 ] && [ "$has_evi" -ge 1 ]; then
          R_FULL=$((R_FULL+1))
        else
          R_MISS=$((R_MISS+1))
        fi
        # 字段3 条款: 逐字抄录须含该条正文实质(不得"见手册第X条"/摘要;须含该条编号本体+非空正文)
        term=$(echo "$ln" | sed 's/.*条款: //;s/| 代码:.*//' | xargs)
        if [ -z "$term" ] || echo "$term" | grep -qE '^见手册|^见第|^同上|^同条款'; then R_TERMMISS=$((R_TERMMISS+1)); fi
        if echo "$term" | grep -cE '[一-龥]{8,}' | grep -q '^0$'; then R_TERMMISS=$((R_TERMMISS+1)); fi
        # 字段4 代码: 绝对定位须含行号(文件::符号:行号 或 文件:行号;防只写文件名)
        if ! echo "$code" | grep -qE '[:：]\s*[0-9]+[–\-]?[0-9]*|第\s*[0-9]+\s*[行条]|行号'; then R_CODELINEMISS=$((R_CODELINEMISS+1)); fi
        # 字段5 落实: 一级字段须含实质(≥3 类实指:路径/符号/参数/断言/分支/写入,防"已落实"空话)
        detail=$(echo "$ln" | sed 's/.*落实: //;s/| 审查:.*//')
        e_path=$(echo "$detail" | grep -cE '\.(py|sh|json|yaml|yml|md|txt)\b|\.py::|/[A-Za-z_]/' || true)
        e_sym=$(echo "$detail" | grep -cE '::|\(\)|键|字段|变量|函数|类' || true)
        e_val=$(echo "$detail" | grep -cE '[0-9]+(\.[0-9]+)?(m|ms|%|Hz|s|GB|epoch|seed)?' || true)
        e_assert=$(echo "$detail" | grep -cE 'assert|断言|校验|检查' || true)
        e_branch=$(echo "$detail" | grep -cE 'if |分支|条件|触发|当 ' || true)
        e_write=$(echo "$detail" | grep -cE '写|返回|输出|入|落盘' || true)
        n_elem=0
        [ "$e_path" -ge 1 ] && n_elem=$((n_elem+1))
        [ "$e_sym" -ge 1 ] && n_elem=$((n_elem+1))
        [ "$e_val" -ge 1 ] && n_elem=$((n_elem+1))
        [ "$e_assert" -ge 1 ] && n_elem=$((n_elem+1))
        [ "$e_branch" -ge 1 ] && n_elem=$((n_elem+1))
        [ "$e_write" -ge 1 ] && n_elem=$((n_elem+1))
        if [ "$n_elem" -lt 3 ]; then R_THIN=$((R_THIN+1)); fi
        # 字段6 审查: 必须含"问题/修复/核对"之一(防"审查通过/没问题"空话)
        rev=$(echo "$ln" | sed 's/.*审查: //;s/| 验证:.*//')
        if ! echo "$rev" | grep -qE '问题|修复|核对'; then R_REVTHIN=$((R_REVTHIN+1)); fi
        # 字段6-轨迹: 经循环修改的审查必须写全"违反→改→重映射→不违反"轨迹(须同时含 重映射+不违反),禁止只写"修复:已改好"无轨迹
        rev_note=$(echo "$ln" | grep -cE '违反' || true)
        if [ "$rev_note" -ge 1 ]; then
          if ! echo "$rev" | grep -qE '重映射' || ! echo "$rev" | grep -qE '不违反'; then R_REVTRACE=$((R_REVTRACE+1)); fi
        fi
        # 字段7 验证: 必须含"命令/预期/→/exit/输出"之一(防"已测试/跑通了")
        ver=$(echo "$ln" | sed 's/.*验证: //;s/| 证据:.*//')
        if ! echo "$ver" | grep -qE '命令|预期|→|exit|输出'; then R_VERTHIN=$((R_VERTHIN+1)); fi
        # 字段8 证据: 必须含路径+行号(文件:行号 或 路径+第N行;防"见日志/已保存"空话、防只写文件名不写行号)
        evi=$(echo "$ln" | sed 's/.*证据: //')
        if ! echo "$evi" | grep -qE '\.|/|log|json|md|txt'; then R_EVIMISS=$((R_EVIMISS+1)); fi
        if ! echo "$evi" | grep -qE '[:：]\s*[0-9]+[–\-]?[0-9]*|第\s*[0-9]+\s*[行条]|行号|:[0-9]+|stdout|落盘|验证输出'; then R_EVIMISS=$((R_EVIMISS+1)); fi
        # 字段6-C6 验证命令真实执行:只读命令白名单+5s超时;exit≠0或输出非空但不含预期 → FAIL;无法执行/输出为空 → 转人工复核
        ver_cmd=$(echo "$ver" | sed 's/→.*//' | xargs)
        ver_exp=$(echo "$ver" | sed 's/.*→//' | xargs)
        if [ -n "$ver_cmd" ] && echo "$ver_cmd" | grep -qE '^(bash|python|grep|head|cat|ls|test|\[|find)'; then
          if echo "$ver_cmd" | grep -qE 'rm |mv |>>|> |sudo|curl |wget '; then
            R_VERRUN_BAD=$((R_VERRUN_BAD+1)); R_VERMANUAL=$((R_VERMANUAL+1))
          else
            if timeout 5 bash -c "$ver_cmd" >/tmp/vs_out.$$ 2>&1; then
              out=$(cat /tmp/vs_out.$$)
              if [ -z "$ver_exp" ] || grep -qF "$ver_exp" /tmp/vs_out.$$; then
                R_VERRUN_OK=$((R_VERRUN_OK+1))
              elif [ -z "$out" ]; then
                R_VERRUN_OK=$((R_VERRUN_OK+1)); R_VERMANUAL=$((R_VERMANUAL+1))  # 退出码0但无输出 → 人工复核
              else
                R_VERRUN_BAD=$((R_VERRUN_BAD+1))
              fi
            else
              R_VERRUN_BAD=$((R_VERRUN_BAD+1))
            fi
            rm -f /tmp/vs_out.$$
          fi
        else
          R_VERMANUAL=$((R_VERMANUAL+1))  # 非白名单命令 → 人工复核
        fi
        # 字段7-C7 证据文件必须真实存在，且文件内能 grep 到代码字段的符号/断言
        evi_file=$(echo "$evi" | grep -oE '[A-Za-z0-9_./-]+\.(py|sh|json|yaml|yml|md|txt|log|npz|csv)' | head -1)
        sym=$(echo "$code" | grep -oE '[A-Za-z_][A-Za-z0-9_]*::[A-Za-z_][A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*\.\(\)|assert [^,;|]*' | head -1)
        if [ -n "$evi_file" ] && [ -f "$evi_file" ]; then
          if [ -z "$sym" ] || grep -qF "$sym" "$evi_file" 2>/dev/null; then
            R_EVIFILE_OK=$((R_EVIFILE_OK+1))
          else
            R_EVIFILE_BAD=$((R_EVIFILE_BAD+1))
          fi
        else
          R_EVIFILE_BAD=$((R_EVIFILE_BAD+1))
        fi
        ;;
    esac
  done < "$REP"
  chk "C1 八字段齐全行($R_FULL条；缺字段$R_MISS条；须含条款/代码/落实/审查/验证/证据全部8字段)" "$([ "$R_MISS" = "0" ] && echo 1 || echo 0)"
  chk "C2 落实说明详实(要素<3类共$R_THIN条；一级字段:须含落点+改动/参数/断言任一实指，防'已落实'空话)" "$([ "$R_THIN" = "0" ] && echo 1 || echo 0)"
  chk "C3 审查字段有实质(无问题/修复/核对共$R_REVTHIN条；防'审查通过'空话)" "$([ "$R_REVTHIN" = "0" ] && echo 1 || echo 0)"
  chk "C3b 循环修改轨迹完整(经循环修改但审查缺'违反→改→重映射→不违反'轨迹共$R_REVTRACE条；防只写'修复:已改好'无轨迹)" "$([ "$R_REVTRACE" = "0" ] && echo 1 || echo 0)"
  chk "C4 验证字段有命令/预期(无实质共$R_VERTHIN条；防'已测试'空话)" "$([ "$R_VERTHIN" = "0" ] && echo 1 || echo 0)"
  chk "C5 证据字段有路径+行号(无路径共$R_EVIMISS条；防'见日志'空话)" "$([ "$R_EVIMISS" = "0" ] && echo 1 || echo 0)"
  chk "C6 验证命令真实执行(白名单真跑通过$R_VERRUN_OK条/失败或预期不符$R_VERRUN_BAD条；防'验证:跑通了'编造)" "$([ "$R_VERRUN_BAD" = "0" ] && echo 1 || echo 0)"
  chk "C7 证据文件真实存在且含符号($R_EVIFILE_OK真/$R_EVIFILE_BAD假；防'证据:见日志'假路径)" "$([ "$R_EVIFILE_BAD" = "0" ] && echo 1 || echo 0)"
else
  if [ "$PHASE" = "phase1" ]; then
    chk "C 报告格式核验(缺,第1阶段必须生成落实报告)" 0
  else
    echo "  ⚠ [phase0] 报告格式核验 C 跳过 —— 主落实报告属第1阶段产物"
  fi
fi
echo ""
# C3b0 第0阶段落实报告:含"违反"的条款审查必须写全 重映射+不违反 轨迹（治理层条款落实更关键,防'修复:已改好'糊弄）
# 独立于主落实报告执行：第0阶段(主报告未生成)也必须核验第0阶段报告,不得因 REP 不存在而跳过
if [ -f "$REP0" ]; then
  R0_TRACE=0
  while IFS= read -r ln; do
    case "$ln" in
      \[*\]*|\[*)
        rev0=$(echo "$ln" | sed 's/.*审查: //;s/| 验证:.*//')
        if echo "$ln" | grep -q '违反' && ! { echo "$rev0" | grep -q '重映射' && echo "$rev0" | grep -q '不违反'; }; then R0_TRACE=$((R0_TRACE+1)); fi
        ;;
    esac
  done < "$REP0"
  chk "C3b0 第0阶段报告循环轨迹完整(缺轨迹$R0_TRACE条；治理层条款经循环修改须写全'违反→改→重映射→不违反')" "$([ "$R0_TRACE" = "0" ] && echo 1 || echo 0)"
else
  echo "  ⚠ 第0阶段落实报告 $REP0 不存在（phase0 必生成,请先落实治理层条款）"
fi
echo ""


echo "========== D. 代码落点核验(证据/代码路径真实存在) =========="
if [ -f "$REP" ]; then
  D_OK=0; D_BAD=0
  while IFS= read -r ln; do
    case "$ln" in
      \[*\]*|\[*)
        code=$(echo "$ln" | sed 's/.*代码: //;s/| 落实:.*//' | xargs)
        file=$(echo "$code" | cut -d: -f1)
        file=$(echo "$file" | sed 's/.*(//;s/).*//')
        [ -n "$file" ] || file="$code"
        if [ -n "$file" ] && [ -e "$file" ]; then D_OK=$((D_OK+1)); else D_BAD=$((D_BAD+1)); fi
        ;;
    esac
  done < "$REP"
  chk "D 代码落点真实存在($D_OK真/$D_BAD假)" "$([ "$D_BAD" = "0" ] && echo 1 || echo 0)"
else
  if [ "$PHASE" = "phase1" ]; then
    chk "D 代码落点核验(缺,第1阶段必须生成落实报告)" 0
  else
    echo "  ⚠ [phase0] 代码落点核验 D 跳过 —— 主落实报告属第1阶段产物"
  fi
fi
echo ""

echo "========== E. 清场核验 =========="
if [ -f "clear_workspace.py" ]; then
  chk "E1 clear_workspace.py 存在" 1
  QV="清场验证.txt"
  QV2="logs/clear_workspace_第0阶段.log"
  if [ -f "$QV" ]; then
    chk "E2 清场验证文件存在" 1
  elif [ -f "$QV2" ]; then
    chk "E2 清场验证文件存在(logs/clear_workspace_第0阶段.log)" 1
  else
    chk "E2 清场验证文件存在(缺,须在启动前生成;可放 清场验证.txt 或 logs/clear_workspace_第0阶段.log)" 0
  fi
else
  chk "E1 clear_workspace.py 存在(缺,第0阶段第一动作)" 0
fi
echo ""

echo "========== 总判定 =========="
if [ "$FAIL" = "0" ]; then
  echo ">>> PASS: 核验全部通过,门禁放行"
else
  echo ">>> FAIL: $FAIL 项未过,禁止启动实验(先补齐)"
fi
exit "$FAIL"
