# ⑱-1 治理层与项目卫生清点 — Decision Log

**时间戳**: 2026-09-07
**执行者**: ZCode Agent (ZCM-1)
**门类型**: Pre-gate (开工前治理清点)

## 治理层文件清点 (⑱-1①)

| 条款 | 检查内容 | 通过 | 留痕证据 |
|---|---|---|---|
| ⑱-1① | AGENTS.md 存在且五铁律逐字一致 | ✓ | `/AGENTS.md` |
| ⑱-1① | pre-commit 钩子存在且断言逐字一致 | ✓ | `/.git/hooks/pre-commit` |
| ⑱-1① | 规则清单与手册编号同步 | ✓ | `/checklists/gate_01_governance.md` |

## 项目卫生清点 (⑱-1②)

| 行动 | 源文件 | 结果 |
|---|---|---|
| 终止僵尸进程 PID 33820 (`python -i`, parent=PowerShell stale REPL, 无 experiment cwd/cmdline) | — | 已 SIGKILL |
| **未清理** (另一体系消螩实验): PID 37524 (`E:\消螩实验\scripts\_run_one_spec.py`) | — | 保留，未越界 |
| **未清理** (本实验激活中): PID 14224 (`_eval_paper.py`) | — | 留作后续 RZ 门核对是否归属本轮次 |
| **未清理** (本实验激活中): PID 41692 (`02_generate_sim_raw.py --sim_e9_ablat`) | — | 留作后续 RZ 门核对 |
| 删除 `lstm_ekf.yaml.bak` | untracked .bak | 已删除 |
| 归档 `_fix_manifest_v2.py` → `archive/_fix_manifest.py` | tracked | git rm --cached |
| 归档 `transformer_ekf_v2.yaml` → `archive/transformer_ekf.yaml` | tracked | git rm --cached |
| 归档 `scripts/19_b2_v3_threshold_scan.py` → `archive/19_b2_threshold_scan.py` | tracked | git rm --cached |
| 归档 `AUDIT_REPORT_20260902.md` → `archive/` | tracked | git rm --cached |
| 归档 `HANDOFF_ROUND3_20260906.md` → `archive/` | tracked | git rm --cached |
| 归档 `configs/models/lstm_ekf_smoke.yaml` | tracked | git rm --cached |
| 归档 `_eval_smoke.log` 等 5 个 smoke log | untracked | mv archive/ |
| 归档 `_p10_rz2_full_train.py` | untracked | mv archive/ |
| `archive/` 加入 `.gitignore` | `.gitignore` | ✓ |
| `lstm_ekf.yaml.bak` 已删除 | `.gitignore` `*.bak` | ✓ |

## 比对前置工序 (⑱-1③)
顺序: 治理层清点 → 项目卫生 → 手册 vs 代码比对 — 依次执行.

## 治理层同步义务 (⑱-1④)
`AGENTS.md` ✓, `pre-commit` ✓, `.gitignore` ✓ 同步更新.

## 两实验治理层隔离 (⑱-1⑤)
本实验工作区 `E:\异步高NLOS`, 另一体系消螩实验 `E:\消螩实验` — 各管其域, 未越界.
