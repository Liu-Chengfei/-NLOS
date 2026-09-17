# Stage1 条款 [17] 落实报告

**条款**: 拆解复核（把大批次拆成可独立勾检的小块并逐块复核，防单线执行干漏）
**Severity**: Hard
**状态**: ✅ PASS

## 1. 条款理解

条款 [17] 要求：
- **拆解复核**：把大批次拆成可独立勾检的小块并逐块复核，防单线执行干漏
- **凡可拆分的检查都要拆**：在每一个批次、每一道 gate、每一类检查（清点/比对/审计/核对/验收）中，都应主动拆分并行的子任务/复核子任务协助执行
- **不是"能并行才并行"，而是"凡可拆分的检查都要拆"**
- **用并行独立进程降低"单线从头做到尾导致干漏"的风险**
- **并行度核算仍受 Pre-5 的 12GB/CPU 闸门约束，任何时刻整机已用内存 ≤ 12GB**

配套使用：执行模式独立检查、三样锚点、必交物核对

## 2. 实现方案

### 2.1 新增模块

**文件**: `src/liquidloc/common/batch_decomposer.py`

**核心类**:
- `SubTask`: 单个子任务（带复核状态）
- `BatchDecomposition`: 批次拆解结果
- `ReviewTracker`: 子任务复核状态跟踪器（防漏核）
- `SubCheckSplitter`: 检查拆分器（4种拆分策略）
- `BatchDecomposer`: 批次拆解器

**核心函数**:
- `execute_parallel_checks()`: 并行执行拆分子检查

### 2.2 拆分策略

| 策略 | 类型 | 说明 | 示例 |
|------|------|------|------|
| 静态检查与运行时断言分开并行 | `static_vs_runtime` | 静态代码检查与运行时断言独立并行 | 代码 lint + 单元测试并行 |
| 多文件比对拆成多路 grep | `multi_file_grep` | 多文件检查拆成多路 grep | 多文件关键词搜索并行 |
| gate 必交物核对与 decision log 撰写并行 | `gate_vs_decision_log` | gate 检查与日志撰写独立并行 | gate 验证 + log 生成并行 |
| 清点与复核者交叉核对并行 | `count_vs_review` | 清点与复核交叉核对 | 任务计数 + 复核检查并行 |

### 2.3 关键设计约束

1. **凡可拆分的检查都要拆**: 4种拆分策略覆盖静态/运行时、多文件、gate/log、清点/复核等场景
2. **逐块复核**: ReviewTracker 跟踪每个子任务的复核状态，防止漏核
3. **并行度 ≤12GB RAM**: BatchDecomposer 接受 ram_limit_gb 参数约束
4. **交叉核对**: cross_review_batch() 实现清点与复核的交叉核对
5. **独立进程**: execute_parallel_checks() 使用 ThreadPoolExecutor 实现并行独立执行

### 2.4 核心数据结构

```
SubTask (子任务)
  ├── task_id: 唯一标识
  ├── parent_batch_id: 父批次ID
  ├── split_type: 拆分类型
  ├── status: 复核状态 (pending/running/completed/failed/reviewed)
  ├── result: 执行结果
  └── reviewed_at: 复核时间戳

BatchDecomposition (批次拆解)
  ├── batch_id: 批次ID
  ├── total_tasks: 子任务总数
  └── sub_tasks: 子任务列表

ReviewTracker (复核跟踪器)
  ├── register_sub_task(): 注册子任务
  ├── mark_completed(): 标记完成
  ├── mark_reviewed(): 标记复核
  ├── verify_all_reviewed(): 验证全部复核
  └── get_summary(): 复核覆盖率统计
```

## 3. 代码变更

### 3.1 新增文件

```
src/liquidloc/common/batch_decomposer.py
```

### 3.2 更新文件

```
src/liquidloc/analysis/clause_gates.py
  - 新增 P17 gate 函数: p17_batch_decomposition()
  - 注册到 _ALL 列表
```

## 4. 验证结果

### 4.1 手动验证

| 验证项 | 结果 |
|--------|------|
| 模块文件存在 | ✅ |
| 所有必需符号存在 (6/6) | ✅ |
| BatchDecomposer 拆解批次 | ✅ |
| 并行子任务执行 | ✅ |
| 交叉核对 (cross_review) | ✅ |
| SubCheckSplitter 4种拆分策略 | ✅ |
| execute_parallel_checks 并行执行 | ✅ |
| ReviewTracker 复核状态跟踪 | ✅ |
| 复核覆盖率 = 100% | ✅ |
| 已完成任务 = 子任务总数 | ✅ |

### 4.2 Gate 验证

```
PASS [P17-decompose] [17] 拆解复核
```

## 5. 与其他条款的关联

- **C-7**: 并行度 = min(GPU, RAM, 待执行) → 拆解复核并行度受此约束
- **C-6**: RAM ≤12GB 硬上限 → 并行执行受此约束
- **[16]**: 资源冲突错峰/排队 → 拆解复核的并行度计算依赖资源冲突调度器
- **[14]**: 并行-串行等效性抽查 → 拆解复核的并行执行结果需等效验证
- **[15]**: 并行调度记录留痕 → 拆解复核的决策同步留痕

## 6. 总结

条款 [17] 已完整落实：
- 新增 `BatchDecomposer`、`ReviewTracker`、`SubCheckSplitter` 模块实现拆解复核
- 4种拆分策略覆盖所有检查类型
- ReviewTracker 确保逐块复核，防止漏核
- 交叉核对机制验证复核完整性
- 并行执行受 RAM ≤12GB 约束
- P17 gate 验证通过
