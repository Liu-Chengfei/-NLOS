---
name: agents-governance-writer
description: 治理写作核心 skill。编写或重写 `AGENTS.md`、repo-local `SKILL.md`、`docs/codex/*` 等治理文档时使用，统一骨架、边界、权威顺序，并保持科学真相层与施工外壳严格分层。
---

# agents-governance-writer

## 1. 使用场景
- 新增或重写根/局部 `AGENTS.md` 时。
- 新增或重写 repo-local `SKILL.md` 时。
- 重写 `docs/codex/*`、治理类文档时。

## 2. 不适用场景
- 不用于写 `SKILL.md` 或科学真相文档。
- 不用于定义实验真相、指标口径或协议边界。

## 3. 输入要求
- 目录职责、允许改动、禁止改动、邻接边界、验证门槛。
- skill 目标、触发场景、边界、预期交付。
- 当前权威源、目录边界、拟写文件列表。

## 4. 输出格式
- 固定 8 段：职责、允许、禁止、权威、风险、验证、边界、停损。
- 写 `SKILL.md` 时固定 8 段：使用场景、不适用、输入、输出、决策顺序、禁止、边界、验证。
- 分层治理结果必须明确：`科学真相层 / 施工外壳层 / 历史层`。

## 5. 决策顺序
1. 先确认科学真相属于 `docs/`、`configs/base/`、`protocol/`。
2. 再确认施工规则属于 `AGENTS.md`、`.agents/skills/`、`docs/codex/`。
3. 若写 `AGENTS.md`，先写目录职责，再写允许/禁止、权威顺序和边界，最后写风险和停损。
4. 若写 `SKILL.md`，先定 skill 只做一件事，再写触发与不触发，最后写输出和禁区。
5. 同一规则只保留一层主定义。

## 6. 禁止事项
- 复制科学规则进 `AGENTS.md`。
- 跨两层重复写同一规则。
- skill 之间大量重叠。

## 7. 与其他 skill 的边界
- 本 skill 同时覆盖 `AGENTS.md`、repo-local `SKILL.md` 与治理文档分层写作。

## 8. 最小验证/复核要求
- 检查是否与父目录冲突、是否只收紧不放宽。
- front matter、名称、边界、输出格式一致。
- 检查引用链和覆盖顺序不冲突。
