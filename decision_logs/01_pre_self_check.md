# ⑮-1 + Pre-1..Pre-5 自检 (开工前) — Decision Log

**时间戳**: 2026-09-07 15:48
**执行者**: ZCode Agent (ZCM-1)
**门类型**: Pre-gate (Pre-1..Pre-5 开工前自检)

## ⑮-1 五问（开工前自检）

| # | 问题 | 答案 |
|---|---|---|
| 1 | 我在执行哪一册手册？ | 异步高NLOS实验，git HEAD = f75586dc 之前的 commit + 当前 d35616f7 (⑱-1 治理层提交) |
| 2 | 当前磁盘 manifest 显示上一批次进度到哪？ | 无(项目为冷启动；.audit/decision_log.json 等历史 audit 文件存在但不代表当前批状态) |
| 3 | 五铁律已加载并复述？ | 已加载：①唯一真源=手册 ②状态只认磁盘 ③原文执行 ④一步一验 ⑤不确定不跑 |
| 4 | 本批次 DAG 路径（起点→终点）？ | ⑱-1 治理 → Pre-1..Pre-6 → 冒烟+预达标 → RZ-0 → RZ-1 → RZ-2 → RZ-3 → A-1..A-9 |
| 5 | 是否确定严格完美执行每一条？ | 否 — 手册 900+ 行尚未整册常驻；本批先完成 Pre-1..Pre-5 + 治理层建档，再逐批精读后续条款；任一不确定处停下重读 |

## Pre-1 环境锁定 + 工具链四件套

| 项 | 手册要求 | 实测 | 通过 | 证据 |
|---|---|---|---|---|
| GPU 型号 | RTX 5060 | NVIDIA GeForce RTX 5060 Laptop GPU | ✓ | nvidia-smi |
| GPU 显存 | 16GB (手册预设) | **8151 MiB (8GB)** | ⚠ 不符 | 报告：手动册预设 16GB；实测 8GB 显存 — 需降并发/精打细算 batch |
| GPU compute_cap | 12.0 (Blackwell) | 12.0 | ✓ | nvidia-smi + torch.cuda.get_device_capability(0) |
| CUDA 驱动 | 兼容 cu12.8/torch 2.12 | 596.49 | ✓ | nvidia-smi |
| nvcc | 已装 | 13.2.78 | ✓ | nvcc --version |
| PyTorch 内置 CUDA | cu12.8 | 12.8 | ✓ | torch.version.cuda |
| cuDNN | 92000 | 92000 | ✓ | torch.backends.cudnn.version() |
| GCC | 已装 | 15.2.0 (MinGW) | ✓ | gcc --version |
| GPU 可用性 | True | True | ✓ | torch.cuda.is_available() |
| 跨机一致性 | 仅在本机跑全量 | 单一设备 | ✓ | 不涉及 |

**关键报告**：手册 Pre-1 设备基线预设 GPU 16GB 显存；本机实测 8GB — 实际并发数与 batch 内存预算需按 8GB 重新核算。

## Pre-2 决策日志 (decision log)

- 路径：`E:\异步高NLOS\decision_logs\`
- 初始化文件：`00_governance_check.md`, `01_pre_self_check.md`
- 协议变更/分析变更/执行偏差三级分类：本批次暂无

## Pre-3 实验目录结构

| 目录 | 用途 | 状态 |
|---|---|---|
| `data/raw` | 仿真原始数据 (S8 产物) | 已存在 (有 seed0..seed9 旧批次) |
| `data/processed/sim_e9_paper` | 处理后 dataset (论文级 sim_e9 混合池) | 已存在 (有 seed0..seed9) |
| `data/processed/test_one` | 单条测试 | 已存在 |
| `checkpoints/paper_full_50unit` | RZ-2 论文级权重 | 已存在 (4 method 训练痕迹) |
| `checkpoints/liquid_seed0/lstm_seed0/transformer_seed0` | 单 seed 冒烟权重 | 已存在 (非论文级) |
| `logs/` | 训练/评估/生成日志 | 已存在 |
| `decision_logs/` | 本批决策记录 | 已创建 |
| `checklists/` | 规则勾选清单 | 已创建 |
| `archive/` | 旧版本/废弃物理隔离 | 已创建 (已 gitignore) |
| `configs/{base,datasets,experiments,models,scene_tasks,sweeps}` | 配置文件 | 已存在 (5 方法 yaml) |

## Pre-4 数据校验和

- 现有 `data/processed/sim_e9_paper/seed0/prepare_manifest.json` (2.1MB)
- 待 RZ-1 重生成时同步 SHA-256

## Pre-5 资源预算与调度

- GPU: RTX 5060 8GB (实测)
- 整机 RAM: 16GB (实测 psutil)
- 训练默认 GPU ✓ (C-4)
- 12GB RAM 闸门: 本机 16GB 充裕，但已用 Python 进程 (此前 14224/41692) 已退出，目前空闲
- 双项目并发: 消融实验进程当前已退出 (or 已停)，本实验独占
- 仲裁通道: 论文级启动前必查 `~/exp_arbiter/status.json` (⑫ 协议)

## 僵尸进程清点 (S5 三步法)

| PID | 状态 | cmdline 归属 | 处置 |
|---|---|---|---|
| 33820 | 旧 | `python -i` (PowerShell stale REPL) | **已 SIGKILL** (上批) |
| 14224 | 旧 | `_eval_paper.py` (本实验) | 已退出 |
| 41692 | 旧 | `02_generate_sim_raw.py` (本实验) | 已退出 |
| 37524 | 旧 | `E:\消融实验\_run_one_spec.py` (另一体系) | 未动 (⑪ 文件隔离) |

**当前状态**：无 Python 进程，无 zombie 进程，GPU 空闲，磁盘空闲 75GB。
