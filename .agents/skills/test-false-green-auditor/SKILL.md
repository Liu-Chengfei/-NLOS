---
name: test-false-green-auditor
description: 专查 skip、过强 monkeypatch、只测 happy path、伪造 artifact 等假绿风险。
---

# test-false-green-auditor

## 1. 使用场景
- 审查 tests 覆盖质量和“通过但没真测到”问题时。

## 2. 不适用场景
- 不替代业务逻辑逐行审查。

## 3. 输入要求
- 目标测试文件、被测路径、mock/monkeypatch 使用点。

## 4. 输出格式
- `假绿点 / 被遮蔽的真实路径 / 风险等级 / 最小补测建议`。

## 5. 决策顺序
1. 先看 skip 和 xfail。
2. 再看 monkeypatch / fake pipeline。
3. 再看是否验证了真正关心的字段。

## 6. 禁止事项
- 因测试太方便就把真实路径全替掉。

## 7. 与其他 skill 的边界
- 验证梯子仍由 `pipeline-validation-ladder` 控制。

## 8. 最小验证/复核要求
- 至少证明一条真实关键路径未被假对象遮住。
