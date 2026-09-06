---
name: device-placement-auditor
description: 设备/序列化风险面核心 skill。仅在任务明确落到 CPU/GPU 分工、训练设备选择、推理默认设备、高层 consumer 误占 GPU、checkpoint 保存/加载、`torch.load`、`map_location` 或 device 元数据时使用；不作为默认入口。
---

# device-placement-auditor

## 1. 使用场景
- 审查训练链、推理链、eval/metrics/plotting 设备分工时。
- 触碰 checkpoint、模型装载、训练报告中的 device 元数据时。

## 2. 不适用场景
- 不替代模型数学正确性审查。
- 不替代默认代码检索、普通 bug 初筛或库文档查询。

## 3. 输入要求
- 设备入口、`auto/cpu/cuda` 规则、quick/full 模式、consumer 列表。
- checkpoint 入口、load/save 点、device 迁移点、相关测试。

## 4. 输出格式
- `GPU 专用 / CPU 默认 / load 路径 / save 路径 / device 语义 / 冲突点 / 最小修复建议`。

## 5. 决策顺序
1. 先锁训练链。
2. 再锁推理和高层 consumer。
3. 再锁 `torch.load(..., map_location="cpu")` 与显式迁移到目标设备。
4. 最后找重复设备决策、隐式装载和冲突。

## 6. 禁止事项
- 因为“可能更快”就扩大 GPU 使用面。
- 隐式按保存时设备直接装载。

## 7. 与其他 skill 的边界
- 本 skill 同时覆盖运行时设备分工与序列化/装载语义。

## 8. 最小验证/复核要求
- 目标 device 测试 + 相关脚本 / pipeline 最小 smoke + checkpoint 相关测试。
