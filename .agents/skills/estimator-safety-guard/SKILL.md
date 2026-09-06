---
name: estimator-safety-guard
description: 处理 estimators、interfaces、fusion 边界时使用，防止模型特例直接侵入几何后端。
---

# estimator-safety-guard

## 1. 使用场景
- 修改 estimator、fusion、model intermediate bridge 时。

## 2. 不适用场景
- 不用于 reader、plotting、文档治理。

## 3. 输入要求
- estimator contract、fusion entry、模型输出 shape、目标测试。

## 4. 输出格式
- 边界检查、侵入点列表、最小修复建议。

## 5. 决策顺序
1. 先锁 estimator state / update contract。
2. 再锁 fusion 只桥接不改真相。
3. 最后看 model intermediate 如何消费。

## 6. 禁止事项
- 让模型输出直接改写协议字段或状态定义。

## 7. 与其他 skill 的边界
- device 相关交给 `device-placement-auditor`。

## 8. 最小验证/复核要求
- 目标 estimator / fusion 测试 + 最小 pipeline smoke。
