---
name: high-risk-line-reviewer
description: 高风险逐线/动态链路核心 skill。只在已经定位到高风险文件后使用，针对 protocol、pipelines、estimators、fusion、factories、场景链路做逐文件逐函数逐段精查，并覆盖 factory、registry、字符串路由、路径拼装、懒加载和 checkpoint 解析链；不作为默认发现入口。
---

# high-risk-line-reviewer

## 1. 使用场景
- 第 8 遍高风险逐行精查，或修完某类后复查时。
- 已经通过 `cocoindex-code`、闭包审查或其他专项 skill 缩到少量高风险文件时。
- 审查 create/load/resolve/path/checkpoint 相关问题时。
- LSTM/Liquid、script/pipeline、reader/mapper/event-builder 存在近似逻辑，需要并排对照时。

## 2. 不适用场景
- 不替代广撒网式闭包发现。
- 不替代默认代码检索或初筛。
- 不替代普通 API 审查。

## 3. 输入要求
- 高风险文件清单、目标问题类、对应上游协议。
- factory、registry、脚本入口、checkpoint 和 output root 入口。
- 需要对照的文件组、预期共享合同、当前差异。

## 4. 输出格式
- `文件 / 函数 / 行块 / 动态入口或共同合同 / 风险类型 / 证据 / 最小修复建议`。

## 5. 决策顺序
1. 先按高风险顺序排序。
2. 再找显式支持表、字符串路由和路径拼装。
3. 再逐函数读并抽共同合同、标允许差异。
4. 最后回看直接调用方和测试。

## 6. 禁止事项
- 只扫函数名不看实现细节。
- 用隐式 fallback 掩盖断链。
- 发现重复后自动发起大抽象。

## 7. 与其他 skill 的边界
- 设备分工与序列化问题交给 `device-placement-auditor`。
- 本 skill 同时覆盖高风险逐线、动态解析链和重复实现对照。

## 8. 最小验证/复核要求
- 逐行结论必须能定位到具体文件和代码块；若涉及重复实现修复，还要复核两边目标测试。
