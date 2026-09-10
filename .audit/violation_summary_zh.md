# 手册-代码逐条核查 · 统一违规汇总清单 (最终修正版)
**版本**：v51final | **日期**：2026-09-09 | **核查范围**：S1-S9 / C1-C10 / CODE1-9 / AUD1-4 / Pre-1-6 / I-1-5 / P1-P39

---

## A. 已修复的静态代码违规（8 类，全部完成）

| # | 条款 | 违规内容 | 散落位置 | 修复动作 | 验证 |
|---|------|----------|----------|----------|------|
| 1 | **CODE-3** | 设备硬编码 `.to("cpu")` / `map_location="cpu"` | `lstm/liquid/transformer trainer.py`, `_materializer.py` 等 | 统一 `DEVICE_CPU` 常量（constants.py 单源） | `grep '.to("cpu")'` = 0 命中 |
| 2 | **CODE-2** | α=0.05 论文级数值散落 | `handbook_p22_statistics.py:156`, `handbook_p_series_tools.py:603/649`, `metrics_quality.py:898` | 新增 `constants.ALPHA=0.05`，全部改引用 | analysis 测试 99 passed |
| 3 | **CODE-2** | W=128 / warmup=10s 散落 | `precheck_orchestrator.py:515/1107-1108`, `train_pipeline.py:2121/2129` | 新增 `WINDOW_SIZE=128` / `WARMUP_S=10.0`，全部改引用 | precheck 测试 105 passed |
| 4 | **C-1/C-2** | `eval_batch_size: 128` → 手册要求"推理 batch=16"；脚本 stale help "=64" | `configs/models/*.yaml`, `scripts/23_train_3nn_paper.py:67`, `scripts/06/07` help 文本 | yaml eval→16, 脚本 default→16, 帮助文本→16, 测试 assertions→16 | **255 passed** |
| 5 | **C-3** | `amp_state["enabled"]` 被解析但从未调用 `torch.autocast`/`GradScaler` —— 训练全程 FP32（声明启用但实际不生效） | `liquid/lstm/transformer trainer.py` 训练循环 | 3 个 trainer 的 `train_one_epoch` 加 `amp_state`+`grad_scaler` 形参；前向包 `with torch.autocast(...)`，反向走 `scaler.scale→unscale_→clip→step→update` | 353 passed |
| 6 | **C-4** | CUDA 不可用时**静默回退 CPU** 训练（手册要求"报错停"） | 3 个 trainer `_resolve_runtime_device` "auto"/"cuda"/"cuda:N" 分支 | 改为 `raise RuntimeError("CUDA unavailable...C-4")` | 模块导入 + 测试通过 |
| 7 | **C-10** | `_try_resume_phase_scheduled_training` 无条件 `load_state_dict` 续跑（手册要求"从头重训、无旧权重 load"） | `liquid/_checkpoint_utils.py` | 加 `if trainer_state["deterministic"] is True: return None`（论文级禁 resume） | 161 passed |
| 8 | **R-6** | 手册定义 R-6 兜底（坚决不降级），`r_series.py` 只实现 R-1~R-5 | `analysis/r_series.py`, `run_handbook_gates.py` | 新增 `R6_check_restart_from_aud1` + 路由；更新 `__all__` | R6 导入 OK |

---

## B. 发现的真实 BUG（代码-手册语义不匹配）

| # | 条款 | 类型 | 问题描述 | 位置 | 修复建议 |
|---|------|------|----------|------|----------|
| BUG-1 | **P5** CODE-5/AUD-1 | 手册-代码语义不匹配 | 手册 P5："轨迹在锚区**凸包内**，距**凸包边界** ≥ 1.5m"；代码实现："距离**最近锚点** < 1.5m" → 两种语义**不等价**（近边非顶点处代码漏判，顶点附近代码误判） | `precheck_orchestrator.py:1050-1051` | 改用"点到凸包最近边距离 ≥ 1.5m"实现 |
| BUG-2 | **Pre-6** 数据-代码规范 | 目录命名不匹配 | P5 实际能跑（从 `seedX/sim_curve_*/gt.json` 读到数据 → 884 violations）；Pre-6 seed 计数 `rglob("sim_meta.json")` 找不到任何（实际是 `seedX/manifest.json`） | `_build_p_checks_cfg:1102-1106` | 统一数据目录命名：`manifest.json` → `sim_meta.json`（或在代码里兼容两者） |

> **诚实说明**：之前误判 P5=884 是"数据问题" —— 其实是代码实现**语义有 bug**（锚距 ≠ 凸包边界距）。修复 BUG-1 后，884 数字会**变化**（因为 P5 语义完全改变）。

---

## C. 数据驱动核查 · 软失败（证据缺失，非违规）

| # | 条款 | 缺失字段 | 原因 |
|---|------|----------|------|
| E1 | P2 | missing_source_stats | 需 S9 派生统计 |
| E2 | P3 | m1_burst_stats | 需 M1 成簇丢包实测 |
| E3 | P12 | train/test nlos stats | 需注入对称统计 |
| E4 | DQ-1~4 | 25-unit 报告 RMSE | 需 RZ-2 全量训练 |

> 这些是**运行时才产出的证据**，裸 cfg 调用必然缺失，属**预期行为**。

---

## D. 合规通过项（静态核查）

| 章节 | 结论 | 证据 |
|------|------|------|
| **S1** 场地 20×20 | ✅ | scene spec |
| **S2** 锚点 A1/A2/A3/A4 | ✅ | block1_gdop_verifier.py |
| **S8** 注入顺序 | ✅ | paper_materializer.py 步骤 |
| **C-1/C-2** batch=16 唯一 | ✅ | yaml + scripts + tests 全同步为 16（255 passed） |
| **C-3** AMP 实际生效 | ✅ | 3 trainer `train_one_epoch` 均加 autocast+GradScaler（原仅解析不生效） |
| **C-4** 无 GPU 报错停 | ✅ | 3 trainer `_resolve_runtime_device` 改 RuntimeError（原静默降 CPU） |
| **C-10** 论文级禁 resume | ✅ | `_try_resume_phase_scheduled_training` 加 deterministic 门控 |
| **R-1~R-6** 归因链 | ✅ | R6_check_restart_from_aud1 补齐 |
| **C-9** 清场脚本 | ✅ | _rz_gates.py RZ-0 |
| **I-1~I-5** 实现清单 | ✅ | 所有 I-gates PASS |
| **AUD-3** 无残留产物 | ✅ | git ls-files 无 .bak/tmp |

---

## E. 核查统计（数据驱动 P1-P39 完整跑）

| 维度 | 通过 | 失败 | 说明 |
|------|------|------|------|
| 静态代码规范 | ✅ | 0 残留 | 3 类散落值已收敛 |
| Pre 门控 | 5/6 | 1（Pre-6 数据目录命名） | 目录规范问题 |
| S9 校验 | 1/5 seed | 4/5 | 现有 sim_e9_paper 非论文级构成 |
| P1-P39 | 25/65 | 40 | 多数证据缺失 + 1 BUG（P5） |

---

## F. 结论与下一步

**代码层面**：CODE-1~5/AUD 静态合规；C-1~C-4/C-10 训练口径全部落地（batch=16、AMP 真生效、无 GPU 报错停、论文级禁 resume）；R-1~R-6 归因链齐全；**P5 边界检查实现有语义 BUG**（需修复凸包距离计算）。

**数据层面**：sim_e9_paper 目录命名 (`manifest.json`) 与 Pre-6 期望 (`sim_meta.json`) 不符 → 需统一命名；S9 仅 1/5 seed 通过。

**运行时层面**：RAM≤12GB 并发（C-6）、并行 max(GPU,RAM,待执行)（C-7）、DQ/G/RZ 全链路需完整 50 单元跑后验证。

> **机器可读明细**：见 `.audit/verification_summary_20260909.json` + `.audit/verification_checklist_zh.md`