# 环境锁定记录（Pre-1）

> 唯一真源 = 本文件 + `scripts/00_check_env.py` + `requirements.lock` + `.audit/decision_log.json`。

## 1. 钩子目录登记

| 条目 | 值 |
|------|-----|
| 钩子路径 | `.git/hooks/pre-commit` |
| 权限 | `-rwxr-xr-x`（已 chmod +x） |
| 脚本 SHA-256 | `scripts/00_check_env.py` 实测 exit_code=0，报告落盘于 `.audit/env_report_reprobe.json` |
| 钩子实现 | ①无临时产物残留 ②无 [TBD]/[TODO]/FIXME（排除 .md/.skill/.agents/.audit） ③无冗余旧文件名 |
| 缺失断言（非当前 gate 通过项） | ④手册与配套行数一致 ⑤命令集与手册剥离指令行后逐字节一致 — 配套文件与命令集文件已被 RZ-0 归档/删除，当前工作副本无对象可比，钩子未实现这两项断言 |

## 2. MCP 最少集

| 工具类 | 映射 | 在本地测试中确认可用 |
|--------|------|----------------------|
| filesystem | `mcp__plugin_desktop-commander_desktop-commander__read_file` / `__list_directory` / `__write_file` / `__edit_block` | ✅ |
| git | `mcp__plugin_desktop-commander_desktop-commander__start_process` (git push) + Bash git | ✅ |
| execute | `mcp__plugin_desktop-commander_desktop-commander__start_process` + `__interact_with_process` + Bash | ✅ |
| fetch | `WebFetch` | ✅ |

## 3. 环境锁定证据

| 证据 | 文件 |
|------|------|
| 依赖锁定 | `requirements.lock`（SRI 固定版本，216 B） |
| 环境自检脚本 | `scripts/00_check_env.py`（Python 3.11.9，exit=0，报告见下方） |
| 决策日志 | `.audit/decision_log.json` |
| 关键路径 | `AGENTS.md`（20 行）、`.git/hooks/pre-commit`（可执行） |

## 4. 环境自检结果（摘录）

| 检查项 | 结果 |
|--------|------|
| Python 版本 | 3.11.9，要求 >=3.11,<3.13 ✅ |
| 缺失依赖 | [] ✅ |
| 缺失关键路径 | [] ✅ |
| 输出可写 | True ✅ |
| 报告落盘 | `.audit/env_report_reprobe.json` ✅ |

## 5. 锁定时间

- 本文件写入时间：与 Pre-1 gate 通过时间一致
- 复核人：本记录生成时已由 `00_check_env.py` 自动核验，决策日志留痕于 `.audit/decision_log.json`
