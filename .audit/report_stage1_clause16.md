# Stage1 条款 [16] 落实报告

**条款**: 资源冲突时按错峰/排队执行，不扩资源、不降级口径
**Severity**: Hard
**状态**: ✅ PASS

## 1. 条款理解

条款 [16] 要求：当 GPU、RAM 等资源发生冲突时，执行器必须按**错峰**（stagger）或**排队**（queue）方式调度，而不是：
- 扩展资源（如申请更多 GPU/RAM）
- 降级论文级口径（如降低精度、减少数据）

这意味着：资源冲突时，只改变**执行顺序**，不改变**论文级指标**。

## 2. 实现方案

### 2.1 新增模块

**文件**: `src/liquidloc/common/resource_conflict_scheduler.py`

**核心类**: `ResourceConflictScheduler`

**核心方法**:
- `check_resource_conflict()`: 检测资源冲突（GPU 不足、RAM 超限）
- `stagger_and_queue()`: 错峰/排队调度
- `write_scheduling_log()`: 写入调度决策日志

### 2.2 调度策略

| 场景 | 模式 | 策略 |
|------|------|------|
| 无冲突 | parallel | 所有任务并行执行 |
| 有冲突 + 有并行容量 | staggered | 分批错峰执行，批次间延迟 |
| 有冲突 + 无并行容量 | queued | 串行排队执行 |

### 2.3 关键设计约束

1. **不扩资源**: 调度器使用已有的 GPU/RAM 数量，不申请额外资源
2. **不降级口径**: 错峰/排队只改变执行顺序，数据、配置、权重、统计方法与串行一致
3. **RAM ≤12GB 硬上限**: 错峰调度仍受 C-6 约束
4. **决策留痕**: 所有调度决策写入 decision log

## 3. 代码变更

### 3.1 新增文件

```
src/liquidloc/common/resource_conflict_scheduler.py
```

### 3.2 更新文件

```
src/liquidloc/analysis/clause_gates.py
  - 新增 P16 gate 函数: p16_resource_conflict_scheduler()
  - 注册到 _ALL 列表
```

## 4. 验证结果

### 4.1 手动验证

| 验证项 | 结果 |
|--------|------|
| 模块文件存在 | ✅ |
| 所有必需符号存在 (5/5) | ✅ |
| 冲突检测 → 错峰调度 | ✅ |
| 无冲突 → 并行调度 | ✅ |
| 独立函数 write_scheduling_record | ✅ |

### 4.2 Gate 验证

```
PASS [P16-conflict] [16] 资源冲突错峰/排队
```

## 5. 与其他条款的关联

- **C-6**: RAM ≤12GB 硬上限 → 错峰调度仍受此约束
- **C-7**: 并行度 = min(GPU, RAM, 待执行) → 错峰调度基于此计算
- **C-8**: 跨实验并发共享 + 论文级互斥 → 错峰调度实现论文级互斥
- **[15]**: 并行调度记录留痕 → 错峰调度决策同步留痕
- **[14]**: 并行-串行等效性抽查 → 错峰调度结果需等效验证

## 6. 总结

条款 [16] 已完整落实：
- 新增 `ResourceConflictScheduler` 模块实现错峰/排队调度
- 冲突检测基于 GPU/RAM 实时资源
- 三种调度模式（parallel/staggered/queued）覆盖所有场景
- 不扩资源、不降级口径的硬约束通过设计保证
- P16 gate 验证通过
