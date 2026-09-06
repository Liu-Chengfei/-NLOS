---
name: data-contract-workflow
description: 数据/协议合同风险面核心 skill。仅在任务明确落到 reader、field mapper、event builder、manifest、split、字段语义保真、protocol、configs/base、输出合同或字段名时使用；不作为默认入口。
---

# data-contract-workflow

## 1. 使用场景
- 修改 `dataio/`、dataset config、样本分流、event 合同时。
- 触碰 `protocol/`、`configs/base/`、结果 schema 或字段名时。

## 2. 不适用场景
- 不用于场景扰动、模型头、metrics 设计。
- 不替代默认代码检索、普通 bug 初筛或库文档查询。

## 3. 输入要求
- 原始字段来源、内部字段合同、样本流向、目标测试。
- 当前协议来源、拟改条目、触发原因、替代方案。

## 4. 输出格式
- `合同检查清单 / 是否真要改协议 / 若不改协议的实现修复路径 / 必要验证`。

## 5. 决策顺序
1. 先锁定原始字段。
2. 再锁内部字段映射。
3. 再锁 `t/dt/source_t/seq_id/scene_id`。
4. 再问是不是实现问题、是不是 consumer / test 影子协议。
5. 最后看 split / manifest / fallback，并仅在确认协议本身错误时才动协议。

## 6. 禁止事项
- 在 reader 注入场景扰动。
- 重命名协议字段去适配局部实现。
- 为了过测或过脚本放宽协议。

## 7. 与其他 skill 的边界
- 场景逻辑交给 `closure-audit-planner` / `high-risk-line-reviewer`。
- 本 skill 同时覆盖数据实现合同与协议改动守门。

## 8. 最小验证/复核要求
- 目标 dataio / protocol 测试 + 直接调用链最小 smoke。
