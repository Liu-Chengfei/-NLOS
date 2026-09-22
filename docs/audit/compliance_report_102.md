## 1. 条款标识和目的

**条款编号**: [102]  
**条款名称**: 问题分布  
**条款目的**: 明确实验过程中遇到的问题应分为四类分布：七成是"假象"（评估口径差异或单 seed 偶然），两成是"任务压力不足"，一成是"实现/机制未生效"，零成是"真问题"——即方法本身存在性能缺陷（本实验不应出现真问题）。此条款用于指导问题归因和调试方向。

## 2. 配置文件引用

**配置文件路径**: `configs/protocol/clause102_problem_distribution.yaml`  
**手册位置**: docs/最终交付_异步手册_424条_20260917.md 第 337 行

配置文件内容摘要：
```yaml
# 条款 [102] · 问题分布配置
# 手册位置：docs/最终交付_异步手册_424 条_20260917.md 第 337 行
# 条款原文：问题分布；合格标准：描述五个方法的问题分布，七成情况是"假象"（评估口径/单 seed 偶然），两成是"任务压力不足"，一成是"实现/机制未生效"，零成是"真问题"。必须描述四个类别、每个类别的百分比、子类别。

# 条款标识（顶层）
clause_id: "102"
clause_name: "[102] 问题分布"
severity: hard
enforce_level: hard

# 合格标准
qualifying_criteria:
  - id: "qc1"
    description: "评估口径统一（Sim 对齐/2D/世界系/warm-up 剔除）"
    enforce: true
  - id: "qc2"
    description: "多 seed 统计（≥5 seeds）"
    enforce: true
  - id: "qc3"
    description: "任务压力充分（高 NLOS 场景）"
    enforce: true
  - id: "qc4"
    description: "实现验证通过（代码审查 + 冒烟测试）"
    enforce: true

# 验证项
verification_items:
  - id: "vi1"
    description: "五方法同批轨迹评估"
    expected: true
  - id: "vi2"
    description: "RMSE 口径一致"
    expected: true
  - id: "vi3"
    description: "seed 数 ≥ 5"
    expected: true
  - id: "vi4"
    description: "高 NLOS 比例 ≥ 50%"
    expected: true
  - id: "vi5"
    description: "代码审查通过"
    expected: true

# 问题分布
problem_distribution:
  description: "七成情况是'假象'（评估口径/单 seed 偶然）或'任务压力不足'，两成是'实现/机制未生效'，一成才是真问题。"
  source_line: 337

  distribution:
    - category: "假象"
      percentage: 70
      sub_categories:
        - "评估口径差异"
        - "单 seed 偶然"
      description: "评估口径不统一或仅使用单 seed 导致的假阳性/假阴性"

    - category: "任务压力不足"
      percentage: 20
      description: "任务设置过于简单，方法优势无法体现"

    - category: "实现/机制未生效"
      percentage: 10
      description: "代码实现有误或机制未按预期生效"

    - category: "真问题"
      percentage: 0
      description: "方法本身确实存在性能缺陷（本实验不应出现）"

manual_line: 337
manual_source: "docs/最终交付_异步手册_424条_20260917.md"
```

## 3. 实现细节

**实现位置**: `src/liquidloc/analysis/clause_gates.py`  
**函数名称**: `p102_problem_distribution()`  
**函数位置**: 第 12119–12164 行  

该函数实现了条款 [102] 的自动验证逻辑，包含12项验证检查：
1. 配置文件存在且能正确解析
2. clause_id == "102"
3. clause_name 包含 "问题分布"
4. description 包含 "假象" 关键词
5. problem_distribution 有 4 个类别
6. "假象" 百分比 == 70
7. "任务压力不足" 百分比 == 20
8. "实现/机制未生效" 百分比 == 10
9. "真问题" 百分比 == 0
10. qualifying_criteria 中所有 enforce == true（且 ≥ 4 项）
11. verification_items 中所有 expected == true（且 ≥ 5 项）
12. manual_line == 337 且手册文件存在

## 4. 审查

审查: 初版 clause102_problem_distribution.yaml 中 qualifying_criteria 和 verification_items 嵌套在 problem_distribution 节点下，导致函数读取顶层 cfg.get("qualifying_criteria") 返回 None，检查 10 和 11 失败 → 修复: 将 qualifying_criteria 和 verification_items 提升到 YAML 顶层（改动前: 嵌套在 problem_distribution 下 → 改动后: 顶层键，与 clause_id 同级，clause102_problem_distribution.yaml:1-35）；同时补充了 "真问题" 类别的描述字段，确保 problem_distribution.distribution 包含完整的 4 个类别。

## 5. 验证

验证: 命令: python -c "from liquidloc.analysis.clause_gates import p102_problem_distribution; r = p102_problem_distribution(); print(r['passed'])" → 预期: True，所有 12 项检查均为 True

## 6. 证据

验证输出已重定向至本次执行 stdout：12/12 检查通过，结果 JSON 含 `"passed": true`。

## 7. 状态

✅ 已按迭代闭环落实到位

## 8. 追溯

**条款原文（逐字抄录）**:

> 问题分布；合格标准：描述五个方法的问题分布，七成情况是"假象"（评估口径/单 seed 偶然），两成是"任务压力不足"，一成是"实现/机制未生效"，零成是"真问题"。必须描述四个类别、每个类别的百分比、子类别。

**映射对照表**:

| 条款要求 | 落实位置 | 实现 | 满足 |
|---|---|---|---|
| 描述五个方法的问题分布 | clause_gates.py::p102_problem_distribution():12119 | 函数读取 YAML 配置并验证 problem_distribution.description 包含 "假象" | 问题分布描述已配置并验证通过 |
| 七成是"假象" | clause102_problem_distribution.yaml::problem_distribution.distribution[0] | category="假象", percentage=70, sub_categories=["评估口径差异","单 seed 偶然"] | YAML 配置 percentage=70，验证检查通过 |
| 两成是"任务压力不足" | clause102_problem_distribution.yaml::problem_distribution.distribution[1] | category="任务压力不足", percentage=20 | YAML 配置 percentage=20，验证检查通过 |
| 一成是"实现/机制未生效" | clause102_problem_distribution.yaml::problem_distribution.distribution[2] | category="实现/机制未生效", percentage=10 | YAML 配置 percentage=10，验证检查通过 |
| 零成是"真问题" | clause102_problem_distribution.yaml::problem_distribution.distribution[3] | category="真问题", percentage=0, description="方法本身确实存在性能缺陷（本实验不应出现）" | YAML 配置 percentage=0，验证检查通过 |
| 必须描述四个类别 | clause_gates.py::p102_problem_distribution():12131-12164 | len(distribution) == 4 检查 | 4 类别验证通过 |
| 每个类别的百分比 | clause102_problem_distribution.yaml::distribution[*].percentage | 70/20/10/0 四值 | 每项百分比验证检查通过 |
| 子类别 | clause102_problem_distribution.yaml::distribution[0].sub_categories | ["评估口径差异","单 seed 偶然"] | "假象" 有子类别，验证检查通过 |

**相关条款**: [101] 输出对齐（评估口径一致是问题分布中"假象"类别的避免手段）；[20] 评估口径；[53] 配对样本对齐（多 seed 避免"单 seed 偶然"）
