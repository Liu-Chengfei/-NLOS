---
name: public-benchmark-guard
description: 处理公开 benchmark、miluv、registry、public pipeline 时使用，防止调参边界漂移和特例逻辑回流。
---

# public-benchmark-guard

## 1. 使用场景
- 修改 public benchmark 链、miluv reader、公开数据路由时。

## 2. 不适用场景
- 不用于私有 experiment config 或纯治理文档。

## 3. 输入要求
- benchmark 入口、数据源、特例来源、目标验证。

## 4. 输出格式
- 风险点、特例隔离策略、最小修复方案。

## 5. 决策顺序
1. 先分清 public 链和通用链。
2. 再检查特例是否回流通用 reader / pipeline。
3. 最后看 artifact 和结论口径。

## 6. 禁止事项
- 把 public benchmark 特例写回通用 reader 或 protocol。

## 7. 与其他 skill 的边界
- 数据字段合同仍由 `data-contract-workflow` 主查。

## 8. 最小验证/复核要求
- public pipeline / registry 相关测试与最小 smoke。
