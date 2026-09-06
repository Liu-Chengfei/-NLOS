---
name: pipeline-validation-ladder
description: 处理 pipelines、scripts 或多层串联任务时使用，控制验证范围，避免一上来跑全仓或全实验。
---

# pipeline-validation-ladder

## 1. 使用场景
- 修改 pipeline、scripts、consumer 链、artifact 合同时。

## 2. 不适用场景
- 不用于决定科学协议。

## 3. 输入要求
- 修改点、调用层级、风险级别、现有测试入口。

## 4. 输出格式
- `目标测试 -> 同层 smoke -> 相关 integration -> 必要最小任务包验证`。

## 5. 决策顺序
1. 先跑目标测试。
2. 再看同层 smoke。
3. 仅在必要时加 integration 和更高层验证。

## 6. 禁止事项
- 局部修复直接跑完整实验矩阵。

## 7. 与其他 skill 的边界
- 不决定修复内容，只决定验证梯子。

## 8. 最小验证/复核要求
- 选择的梯子必须能解释到每个改动点。
