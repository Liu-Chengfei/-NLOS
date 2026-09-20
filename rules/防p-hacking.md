# 防p-hacking（治理层·硬约束）

## 核心判据

禁止"补跑只挑结果好的单元"（选择性补跑 = 变相p-hacking）；禁止"把数据改到结果满意为止"（若属档位/注入语义错误 → 走协议变更流程或缩小声明范围，不得改数据到满意）；修复后结果不得与旧结果混用（纪律）；任何旧数据集一律不得复用（哪怕旧数据本身合格），论文级结论必须来自这批复现可查的全新数据；诊断轨/冒烟数据的任何结果不得以任何形式并入、替代或"补充"验收轨数据（用后即弃或隔离归档，禁止并入任何正式数据）。

## 执行保障

- `src/liquidloc/dataio/manifests/dataset_checks.py`：旧数据复用检查
- `src/liquidloc/analysis/significance_tests.py`：混用检测
- `configs/protocol/clause14_p_hacking_prevention.yaml`：防p-hacking配置
- `src/liquidloc/analysis/clause_gates.py`：p14_p_hacking_prevention()验证函数
- `.git/hooks/pre-commit`：防p-hacking检查调用点
- `AGENTS.md`：治理载体索引[14]条目
- `outputs/compliance_report_14.md`：八字段落实报告
