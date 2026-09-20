#!/usr/bin/env bash
# verify_l3.sh —— L3 语义级人工抽查支持系统
# 用法:
#   bash verify_l3.sh <执行手册.md> [落实报告.md] [抽样数N]
#       从手册(或落实报告)按风险加权抽样 N 条,生成 L3 人工抽查卡
#   bash verify_l3.sh -audit <抽查卡.md>
#       读取人工判定的抽查卡,汇总 PASS/FAIL 结论
# 风险加权: 治理条款(+3) | 超长条款≥200字(+2) | 含禁止/不得/严禁(+2) | 含必须(+1) | 含gate/门禁(+1)
# 抽样: 分数降序取前 N/2,剩余随机补足(N 默认 10,上限 30)
set -u

MODE="${1:?用法: bash verify_l3.sh <手册.md> [落实报告.md] [N] 或 -audit <抽查卡.md>}"
if [ "$MODE" = "-audit" ]; then
  CARD="${2:?需要抽查卡路径}"
  [ -f "$CARD" ] || { echo "✗ 抽查卡不存在: $CARD"; exit 1; }
  TOT=$(grep -c '^## \[' "$CARD" 2>/dev/null || echo 0)
  PASS=$(grep -A3 '人工判定' "$CARD" | grep -ciE '\- \[x\] *PASS' || echo 0)
  FAIL=$(grep -A3 '人工判定' "$CARD" | grep -ciE '\- \[x\] *FAIL' || echo 0)
  UNDEC=$(grep -A3 '人工判定' "$CARD" | grep -cE '\- \[ \] *(PASS|FAIL)' || echo 0)
  echo "========== L3 人工抽检汇总 =========="
  echo "抽查条款总数: ${TOT:-0}"
  echo "  判定 PASS : $PASS"
  echo "  判定 FAIL : $FAIL"
  echo "  未判定     : $UNDEC"
  echo ""
  if [ "$FAIL" -gt 0 ]; then
    echo ">>> FAIL: $FAIL 条未通过,下列条款需回退重做:"
    awk '/^## \[/{cur=$0} /- \[x\] FAIL/{print "  " cur "  → 原因见抽查卡"}' "$CARD"
    exit 1
  elif [ "$UNDEC" -gt 0 ]; then
    echo ">>> 未完成: 还有 $UNDEC 条未判定,补齐判定后重跑"
    exit 2
  else
    echo ">>> PASS: L3 人工抽检全部通过($PASS 条真实落实),可进入下一阶段"
    exit 0
  fi
fi

MANUAL="$MODE"
[ -f "$MANUAL" ] || { echo "✗ 手册不存在: $MANUAL"; exit 1; }
REP="${2:-}"
N="${3:-10}"
[ "$N" -gt 30 ] && N=30
[ "$N" -lt 1 ] && N=1

# ---- 1. 提取条款清单(单行: 条款头行) ----
CLAUSES=$(mktemp)
if [ -n "$REP" ] && [ -f "$REP" ]; then
  echo ">>> 抽样源: 落实报告(已落实条款) $REP"
  grep -E '^\['"'"'[0-9]*'"'"'\]|^\[[0-9]*\]' "$REP" | grep '✅' > "$CLAUSES"
  SRCTAG="落实报告"
else
  echo ">>> 抽样源: 手册全量条款(未给落实报告,抽查卡需人工对照工作区核实) $MANUAL"
  grep -E '^\*\*\[[0-9]+\]\*\*' "$MANUAL" > "$CLAUSES"
  SRCTAG="手册全量"
fi
TOTAL=$(wc -l < "$CLAUSES")
if [ "$TOTAL" -eq 0 ]; then
  echo "✗ 未从抽样源提取到任何条款/落实记录(检查报告格式: 行首 [编号] ✅ | ...)"
  rm -f "$CLAUSES"; exit 1
fi
[ "$N" -gt "$TOTAL" ] && N=$TOTAL

# ---- 2. 风险加权评分 + 抽样 ----
# 输出: 分数\t条款行(用 SOH 分隔避免制表符冲突)
SCORED=$(mktemp)
awk -v TOTAL="$TOTAL" '
function score(l,   s, len, body) {
  s = 0
  # 治理条款: 编号 <= 23
  if (match(l, /^\*\*\[([0-9]+)\]\*\*/, m)) { if (m[1]+0 <= 23) s += 3 }
  # 超长条款 >= 200 字符
  len = length(l)
  if (len >= 200) s += 2
  # 含禁止/不得/严禁
  if (l ~ /禁止|不得|严禁/) s += 2
  # 含必须
  if (l ~ /必须/) s += 1
  # 含 gate/门禁
  if (l ~ /gate|门禁/) s += 1
  return s
}
{ print score($0) "\t" NR "\t" $0 }
' "$CLAUSES" | sort -t$'\t' -k1,1nr -k2,2n > "$SCORED"

# 取 top 前 N/2(至少1条)
HALF=$(( (N+1)/2 ))
TOP=$(head -n "$HALF" "$SCORED")
# 随机补足: 从剩余中固定 seed 抽 (N-HALF) 条
REST=$(tail -n +$((HALF+1)) "$SCORED")
NEED=$(( N - HALF ))
if [ "$NEED" -gt 0 ] && [ -n "$REST" ]; then
  RAND=$(printf '%s\n' "$REST" | awk -v need="$NEED" 'BEGIN{srand(20260919)} {a[NR]=$0} END{n=NR; for(i=0;i<need && i<n;i++){j=int(rand()*n)+1; if(!(j in used)){print a[j]; used[j]=1}}}')
  SELECTED=$(printf '%s\n%s\n' "$TOP" "$RAND")
else
  SELECTED="$TOP"
fi

# ---- 3. 生成抽查卡 ----
OUT="${MANUAL%.md}-L3抽查卡.md"
{
  echo "# L3 人工抽查卡"
  echo ""
  echo "- 抽样源: $SRCTAG"
  echo "- 手册: $MANUAL"
  echo "- 抽样: $N 条 / 共 $TOTAL 条"
  echo "- 抽样策略: 风险加权(治理条款/超长条款/含禁止项 + 随机补足)"
  echo "- 生成时间: $(date '+%Y-%m-%d %H:%M')"
  echo ""
  echo "> 用法: 逐条打开【核对清单】勾选,然后在【人工判定】勾 PASS/FAIL 并写原因;全部判定后运行"
  echo "> \`bash verify_l3.sh -audit $OUT\` 汇总结论。"
  echo ""
  echo "---"
  echo ""
  CNT=0
  printf '%s\n' "$SELECTED" | sort -t$'\t' -k2,2n | while IFS=$'\t' read -r score line no body; do
    CNT=$((CNT+1))
    # 提取条款号与标题
    NUM=$(printf '%s' "$body" | sed -E 's/^\*\*\[([0-9]+)\]\*\*.*/\1/')
    TITLE=$(printf '%s' "$body" | sed -E 's/^\*\*\[[0-9]+\]\*\*\*\*[ ]*\*\*([^*]+)\*\*.*/\1/')
    echo "## [$NUM] ${TITLE}"
    echo ""
    echo "### 条款原文(逐字)"
    echo "> ${body}"
    echo ""
    # 若抽样源是落实报告,报告行里含八字段,原样贴
    if [ "$SRCTAG" = "落实报告" ]; then
      echo "### 落实报告(声称已落实)"
      echo "> ${body}"
      echo ""
    fi
    echo "### 核对清单(逐项勾选)"
    echo "- [ ] **落点真实**: \`代码:\` 字段的 文件/符号/行号 确实存在于工作区(可跑 verify_slim 复核)"
    echo "- [ ] **逐句映射**: 落实: 字段的 ①/②/③… 与条款原文每一句/每一项一一对应,无漏句、无省略、无改写"
    # 提取禁止项
    if printf '%s' "$body" | grep -q '禁止\|不得\|严禁'; then
      echo "- [ ] **禁止项明确**: 条款含禁止项(\"$(printf '%s' "$body" | grep -oE '[^。；;]*禁止[^。；;]*|[^。；;]*不得[^。；;]*|[^。；;]*严禁[^。；;]*' | head -3 | tr '\n' ';')\"),落实: 字段是否明确写出\"未违反\"及对应检查"
    fi
    echo "- [ ] **审查轨迹**: 审查: 字段写了\"问题→修复\"(循环修改的写全 违反→改→重映射→不违反 轨迹)"
    echo "- [ ] **验证可复现**: 验证: 字段的命令能跑、预期输出明确"
    echo "- [ ] **证据真实**: 证据: 字段的路径真实存在"
    echo ""
    echo "### 人工判定"
    echo "- [ ] PASS — 落实真实且完整,备注:"
    echo "- [ ] FAIL — 原因:"
    echo ""
    echo "---"
    echo ""
  done
  # 兜底: 若 while 子壳导致 CNT 未输出,这里补统计
} > "$OUT"

rm -f "$CLAUSES" "$SCORED"
echo "✅ L3 抽查卡已生成: $OUT"
echo "   共 $N 条,请人工逐条核对后在【人工判定】勾选,然后运行:"
echo "   bash verify_l3.sh -audit $OUT"
