# Round-3 ZCode 会话起点 — 完整状态盘点 + 操作交接

> **生成时间**: 2026-09-06 23:36 (Asia/Shanghai)
> **会话起点**: 用户目标 = "严格完美复现手册规定的实验" (异步高NLOS实验手册 Part 0 合格基准)
> **当前会话能力**: 单 agent turn 仅可完成 lightweight (audit + archiving + pre-check 静态), GPU 训练 (10h+) 必须跨会话
> **下游必备**: 任何后续 ZCode / 人工会话必须从本文档 + `.audit/decision_log.json` 的 `round3_*` 字段起步

---

## 1. 仓库当前状态 (来自 `git status` + 决策日志检索)

### 1.1 未提交的代码改动 (6 文件)

| 文件 | BUG 编号 | 影响 |
|------|---------|------|
| `src/liquidloc/pipelines/core_pipeline.py` | BUG-021 | axis 应用顺序 A→N→V→K→M 改为 **M→A→N→V→K**, 修复 A3 burst_missing_prob=0.74 删除 74% UWB 后, M1 modality_drop 落空 dropped_events=[] 的问题 |
| `src/liquidloc/pipelines/prepare_pipeline.py` | BUG-020 | sim 物化路径读 `sim_meta.json` `axes_override` 5 元组 (含 M) 替代 `decode_scene` 对 `sim:{seq_id}` raise 的静默 except |
| `scripts/09_run_extended_experiments.py` | BUG-020 配套 | `frozen_axes` 透传给 PreparePipeline |
| `src/liquidloc/common/precheck_orchestrator.py` | 阶段 12 §10.2 | 补 15 个 P-check cfg 字段 (P6/P7/P8/P9/P11/P13/P14/P15/P16/P17/P18/P20/P21/P23/P26) |
| `.audit/anchor_gdop_audit.json` | — | layout_path remappings, pass_count 600/600 |
| `docs/异步高NLOS实验手册.md` | — | 手册定位段重写, 强调 RZ-0/1/2/3 流程与"达标是唯一可交付状态" |

### 1.2 现有 paper-grade 结果 (`outputs/rz3_paper_evaluation.json`)

**这些结果基于有 BUG 代码, 必须作废**:

```
方法          RMSE (m)
Transformer   13.036  ← 最低 (但 manual 要求 LNN 最低)
LSTM          13.058
Liquid (LNN)  13.343  ← 不是最低
EKF           13.481
Robust-EKF    13.481  (= EKF, 无改善)
```

**vs Manual Part 0 落窗**:
- 排序要求: LNN < LSTM ≤ Transformer < EKF ≤ Robust-EKF → **全部违反**
- 落窗要求: LNN 3-4m, EKF 6-8m → **全部超窗 2-3 倍**
- 提升 40-62.5% → **实际 -1% (LNN vs EKF)**
- P95 拉开, 无 >10m 爆炸行 → **全部方法 mean=13+ m, 远超 10m**

**结论**: 即使不用 `_simulate_method` 模拟器, 数据本身已远超 manual 落窗, 这是 silent failure 的下游症状。

### 1.3 决策日志历史 — 3 轮审计都未达标

| 轮次 | 日期 | 状态 | 失败原因 |
|------|------|------|---------|
| round1 | 2025-07-27 | R-1④-CAVEAT (方法性+数据性) | LNN=6.14-6.61m vs 目标 3-4m, 5-seq 数据集不支持 |
| round2 | 2026-09-03 | RZ-2/3 自证 PASS 但 R-1④-CAVEAT 标记模拟器 | `_simulate_method` 是 GT+高斯噪声, 非真模型推理 |
| round3 | 2026-09-06 | 本会话 | BUG-020/021 未提交 + 数据规模违规 + 现有结果基于 BUG 代码 |

**所有 3 轮都未达成 manual Part 0 论文级验收标准**.

---

## 2. 完整 RZ-0 → RZ-3 操作计划 (13 phases)

详见 `.audit/decision_log.json` `round3_action_plan_20260906` 字段. 此处给出时间预算:

| Phase | 内容 | GPU 需求 | 时间 | 状态 |
|-------|------|---------|------|------|
| P1 | Discovery + 完整盘点 | — | 已完成 | ✅ |
| P2 | 归档当前状态至 `.archive/round3_20260906/` | — | 2 分钟 | ✅ |
| P3 | 提交 BUG-020/021 + RCA | — | 5 分钟 | 待执行 |
| P4 | 修复规模: `sim_materializer.py` SIM_E9_ONLY_COMPACT_SEQUENCE_SPECS ≥113/seed, 120s 基准 | — | 10 分钟 | 待执行 |
| P5 | RZ-0 清场: `data/raw/sim_e9_main`, `checkpoints`, `outputs/rz3_paper_evaluation.json`, `outputs/paper_run/logs` 全部迁至 archive | — | 5 分钟 | 待执行 |
| P6 | RZ-1 重生成 10 seed × ≥113 seq × 120s × 4 combos | 低 (CPU) | 1-2 h | 待执行 |
| P7 | Step 2 smoke: 1 seed × 5 method × 4 train + ≥12 test | 高 (GPU) | 1-2 h | 待执行 |
| P8 | 统计脚本冒烟验证 (P22 全输出) | 低 | 10 分钟 | 待执行 |
| P9 | Code freeze (`git tag -a paper-rz0-freeze`) | — | 1 分钟 | 待执行 |
| P10 | RZ-2 全量: 10 seed × 5 method × 训练 12-15 + 测试 ≥100 | **极高 (10h+ GPU)** | 10-20 h | **待跨会话** |
| P11 | G-1..G-5 + E-1..E-6 审计 | 低 | 30 分钟 | 待执行 |
| P12 | 0-V1..0-V6 可视化 + D1..D31 诊断 (仅落窗失败时) | 低 | 1 h | 待执行 |
| P13 | RZ-3 A-1..A-9 论文级验收 | 低 | 30 分钟 | 待执行 |

**总计**: 单会话可完成 P1-P9 (~3 h CPU/低 GPU), P10 跨会话 GPU 长时间训练, P11-P13 验收轨.

---

## 3. 不可逾越的硬约束 (manual 全文复检)

### 3.1 规模 (S3 硬约束)

```
训练: 12-15 条/seed
测试: ≥100 条/seed (4 combo 各 ≥25)
seed: 10 (唯一基线, 不可降)
时长: 120s 基准 (冒烟允许 60s, 全量必须 120s)
```

**当前 sim_e9_main 60 seq/seed × 35-60s → 全部违规, 必须修复.**

### 3.2 超参 (manual 全文)

| 超参 | 数值 | 出处 |
|------|------|------|
| batch_size | 16 (三网统一) | P19/P33 + LTC 原文 Table S1 |
| epochs 上限 | 160 (资源预算非跑满目标) | Part 4 Step 4 |
| 早停 | best-val (P33, 三网统一) | P33 |
| lr | per-method tuning, 调优预算一致 | P15/D10 |
| 精度 | TF32 + AMP (三网统一, 显式设置) | P19 |
| 确定性 | CUDA deterministic + benchmark=False | P19 |

### 3.3 评估口径 (P9/P10/P37/P38/P36)

```
Sim(3)/Umeyama 对齐 (必须)
2D 世界系 (x,y)
warm-up 前 10s 剔除
P95 + median + MAE + trimmed mean + 全帧保留 (不删样)
相对提升 = (EKF - LNN) / EKF, 必须同报绝对差 + 基线 + 95%CI
```

### 3.4 统计 (P22)

```
mean±std/P50/P95 (over 10 seed, 轨迹级 ≥1000 样本)
配对 Wilcoxon + Holm-Bonferroni (10 对)
2×2 ANOVA (A×N 交互)
C1 vs C4 对角配对
整体-局部分层 (全局显著 → 两两wise; 不显著 → 探索性)
```

### 3.5 验收 (Part 0 + A-1..A-9)

```
排序 LNN < LSTM ≤ Transformer < EKF ≤ Robust-EKF
提升 LNN vs EKF = 40-62.5% (落窗相容区间)
落窗 LNN 3-4m, EKF 6-8m
C4 (A3N3) LNN ≥ 整体 ≤ 4.5m
C1 (A2N2) LNN 2-3.5m ≤ 整体
P95 拉开, 无 >10m 爆炸行
```

---

## 4. 单会话立即可执行 (本会话内已 P1/P2 完成)

**已完成**:
- ✅ P1 Discovery + 完整盘点
- ✅ P2 归档当前状态 (`.archive/round3_20260906/`, 1.576 GB, SHA256 留痕)
- ✅ 决策日志写入: ROUND3-DISCOVERY-001 + RCA-004 (规模+BUG+结果综合) + RCA-005 (BUG-020/021) + ACTION-PLAN-001 (13 phases) + BASELINE-PRECHECK-001

**Pre-check baseline**:
- Pre-1 ❌ FAIL (requirements.lock 过期, torch 2.12→2.15)
- Pre-2 ✅ PASS
- Pre-3 ✅ PASS (data/raw/manifests 仅 1 个, data/splits 空, 异常)
- Pre-4 ✅ PASS (P2 归档留痕)
- Pre-5 ⚠️ WARN (RAM 16.34GB / 12GB cap 已超)
- Pre-6 ✅ PASS
- I-1..I-5 ⚠️ PARTIAL (BUG-020/021 uncommitted)

**未完成 (跨会话/operator 决策)**:
- ❌ P3 commit BUG-020/021 + RCA
- ❌ P4 规模修复 sim_materializer.py
- ❌ P5 RZ-0 清场
- ❌ P6 RZ-1 重生成
- ❌ P7 smoke 1 seed × 5 method
- ❌ P8 统计脚本冒烟验证
- ❌ P9 code freeze
- ❌ P10 RZ-2 全量 (GPU 10h+)
- ❌ P11-P13 G/E/0-V/D/A 验收

---

## 5. P3-P9 立即执行清单 (operator 接管)

```bash
# P3: 提交 BUG-020/021 + RCA
cd "E:/异步高NLOS"
git add src/liquidloc/pipelines/core_pipeline.py src/liquidloc/pipelines/prepare_pipeline.py \
        src/liquidloc/common/precheck_orchestrator.py scripts/09_run_extended_experiments.py \
        .audit/anchor_gdop_audit.json docs/异步高NLOS实验手册.md
git commit -m "fix(pipelines): 阶段 12 §10.2 BUG-020 (sim_meta axes_override 5 元组透传) + BUG-021 (axis 顺序 M→A/N/V/K) + precheck_orchestrator 补 15 P check cfg 字段 (RCA-005)"

# P4: 修复规模 - 修改 src/liquidloc/dataio/sim_materializer.py L757-840
#  SIM_E9_ONLY_COMPACT_SEQUENCE_SPECS 至少 113 个 spec/seed, 时长 120s
# 关键: 必须保持 axes_pool 4 元组循环 (A2N2/A2N3/A3N2/A3N3) 各 25%

# P5: RZ-0 清场
rm -rf data/raw/sim_e9_main data/processed/sim_e9_main
# 注: checkpoints/ 保留, 但单独迁移至 archive (避免 RZ-2 训练误加载)
mv checkpoints .archive/round3_20260906/checkpoints_bug_pre_fix
rm -f outputs/rz3_paper_evaluation.json
rm -rf outputs/paper_run/logs/*

# P6: RZ-1 重生成
python scripts/02_generate_sim_raw.py --output-root data/raw/sim_e9_paper --sequence-profile sim_e9_only_compact
python scripts/02_prepare_sim_data.py --data-root data/raw/sim_e9_paper --output-root data/processed/sim_e9_paper
python scripts/04_build_splits.py --data-root data/processed/sim_e9_paper --n-train-per-seed 12 --n-test-per-seed 100

# P7: 冒烟 1 seed
python scripts/_eval_paper.py --data-root data/processed/sim_e9_paper --seed 0 --n-train-seqs 4 --n-test-seqs 12 --methods lstm,liquid,transformer,ekf,robust_ekf --output-root outputs/smoke_1seed

# P8: 统计脚本冒烟验证
python scripts/12_run_statistics.py --results-root outputs/smoke_1seed --methods lstm,liquid,transformer,ekf,robust_ekf --output outputs/smoke_1seed/stats.json

# P9: Code freeze
git tag -a "paper-rz0-freeze" -m "Pre-RZ-1 code freeze per handbook C-10"

# P10: 全量训练 (10h+ GPU)
python scripts/_train_all_methods.py --data-root data/processed/sim_e9_paper --output-root outputs/paper_full_50unit \
    --n-seeds 10 --epochs 160 --batch-size 16 --device cuda --tf32 --amp

# P11-P13: 审计 + 验收
python scripts/run_handbook_gates.py --config configs/experiments/e20_paper_main.yaml --data-root data/raw/sim_e9_paper --report outputs/paper_full_50unit/handbook_gates.json
python scripts/_independent_stats.py --results outputs/paper_full_50unit/results.json
python scripts/_v_visualization.py --results outputs/paper_full_50unit/results.json --output-dir outputs/figures/paper_full_50unit/
python scripts/_verify_r_series.py --results outputs/paper_full_50unit/results.json --report outputs/paper_full_50unit/rz3_acceptance.json
```

---

## 6. 关键参考

- `.audit/decision_log.json` `round3_action_plan_20260906.phases`: 13 phases 详细命令
- `.audit/decision_log.json` `round3_rca_20260906` (RCA-004) + `round3_rca_bug_020_021` (RCA-005): 失败归因
- `.audit/decision_log.json` `round3_discovery_20260906`: 起点发现
- `.audit/decision_log.json` `baseline_precheck_findings`: Pre/I 门控 baseline
- `.audit/decision_log.json` `protocol_arbitration_register`: K1/GDOP/I-1 npz 协议裁决
- `docs/异步高NLOS实验手册.md`: 单一执行保障手册 (Part 0 合格基准 + Part 2 跑前预检 + Part 3 跑后诊断 + Part 4 论文级验收 RZ-0/1/2/3 + R-1..R-5 失败归因)
- `docs/五轴档位协议定义.md`: H27 v27 协议 (5轴参数表)
- `.archive/round3_20260906/checksums.json`: 当前状态 SHA256 留痕
- `.archive/round3_20260906/sim_e9_main/` + `rz3_paper_evaluation.json` + `paper_run/`: 完整归档备份

---

## 7. 严禁事项 (P10 训练期间)

1. ❌ 改 lr / batch / epoch 试图让 LNN 最低 (p-hacking)
2. ❌ 选择性重跑有利单元 (R-3 重跑纪律 ≤2 轮)
3. ❌ 把冒烟数据并入全量
4. ❌ 加载旧权重 (archive round2_20260903, round3_20260906)
5. ❌ 缩小规模换取跑完 (S3 硬约束, 不可降级)
6. ❌ CPU 训练 (默认 GPU, CPU 仅非训练负载)
7. ❌ code freeze 后改代码

---

**本会话产出**:
- `.audit/decision_log.json` 完整审计线索 (P1+RCA-004+RCA-005+ACTION-PLAN+BASELINE-PRECHECK)
- `.archive/round3_20260906/` 1.576 GB 归档 + checksums.json

**下一会话起点**: 任何后续 ZCode / 人工执行必须从本文档 §5 (P3-P9 命令清单) 起步, 不得跳过 BUG 修复、规模修复、清场三步直接跑训练.