---
name: config-surface-drift-auditor
description: 配置/消费者合同风险面核心 skill。仅在任务明确落到配置链、输出面、artifact layout、summary、analysis、plotting 或 verify 消费链时使用，审查配置默认值、显式覆盖、实验入口、协议绑定和消费者合同；不作为通用代码审查入口。
---

# config-surface-drift-auditor

## 1. 使用场景
- 审查 `configs/`、CLI 参数、pipeline 默认值、experiment/model config 时。
- 审查输出目录、字段名、report、summary、figure、verify 链时。

## 2. 不适用场景
- 不替代动态加载或 consumer 审查。
- 不替代 metrics 公式审查。
- 不替代默认代码检索、库文档查询或普通 bug 初筛。

## 3. 输入要求
- 配置文件、CLI 参数、实际读取点、默认值。
- producer 输出面、consumer 入口、artifact 根目录。

## 4. 输出格式
- `声明位置 / 读取位置 / 默认值 / 覆盖顺序 / 漂移点 / producer -> artifact -> consumer -> verify`。

## 5. 决策顺序
1. 先锁声明。
2. 再锁读取和 merge。
3. 再找 producer 实际输出与 consumer 预期读取。
4. 最后看 verify / report 是否一致，以及 tests 是否只测一条路径。

## 6. 禁止事项
- 让代码里影子默认覆盖配置声明。
- 只看文件存在，不看字段和索引关系。

## 7. 与其他 skill 的边界
- 闭包范围由 `closure-audit-planner` 定义。
- 本 skill 同时覆盖配置面和消费者合同面。

## 8. 最小验证/复核要求
- 配置加载链检查 + 至少核对一个 producer 样本和一个 consumer 读取点 + 目标脚本 / pipeline 测试。
