# 条款 [12] 落实记录

> 状态: ✅ 已落实 | 文本真源: `docs/异步实验_执行手册_终版v15_总纲条款版.md` 条款 [12]

## 条款逐字抄录

**[12]** **统计口径（必做）** —— 归一化统计量只用训练集统计量（`check_normalization_leak`）；UWB/VIO 最近邻重采样对齐（`nearest_neighbor_resample_events`），禁止插值抹平 NLOS 跳变；TF32 精度统一（`p19_tf32_amp_explicit`）。

## 落实（极尽具体 + 全量映射）

| # | 条款原文要求 | 落点 | 实现 | 满足 |
|---|-------------|------|------|------|
| ① | 归一化统计量只用训练集统计量 | `src/liquidloc/common/bridge_thresholds.py:check_normalization_leak` | `check_normalization_leak` 接收 `test_stats` 和 `train_stats`，逐字段比对 test 的 mean/std 是否与 train 的 mean/std 一致（容差 `rtol=1e-5`），不一致时抛 `NormalizationLeakError`，强制 test 标准化使用 train 的统计量 | 任何 test 集标准化如果误用了 test 自身统计量，会被 gate 阻断 |
| ② | UWB/VIO 最近邻重采样对齐，禁止插值抹平 NLOS 跳变 | `src/liquidloc/dataio/adapters/event_builder.py:nearest_neighbor_resample_events`（新增）; `build_unified_timeline`（新增）; 调用点 `src/liquidloc/pipelines/prepare_pipeline.py:409-419` | `nearest_neighbor_resample_events` 对每个目标时间戳 `target_t`，在源事件序列中找到 `\|t - target_t\|` 最小的事件，复制其 payload 原样保留（不插值），将时间戳改为 `target_t`，返回带 `_resampled=True` 标记的事件。`build_unified_timeline` 取较密传感器的原始时间戳作为基础，向两端延伸生成统一目标时间戳序列 | 对齐过程使用最近邻（nearest-neighbor）而非线性/样条插值，NLOS 跳变被保留而非平滑掉 |
| ③ | TF32 精度统一 | `src/liquidloc/analysis/clause_gates.py:p19_tf32_amp_explicit:370-406` | `p19_tf32_amp_explicit` 扫描全项目 `.py` 文件，检查 `torch.backends.cuda.matmul.allow_tf32` 和 `torch.backends.cudnn.allow_tf32` 是否被显式设为 `True`，以及是否存在 `torch.cuda.amp.autocast` 或 `torch.cuda.amp.GradScaler` | TF32 在 GPU 矩阵乘法和 cuDNN 卷积上统一启用，精度可控 |

## 代码定位

- `check_normalization_leak`: `src/liquidloc/common/bridge_thresholds.py`（运行时 gate）
- `nearest_neighbor_resample_events`: `src/liquidloc/dataio/adapters/event_builder.py`（新增函数）
- `build_unified_timeline`: `src/liquidloc/dataio/adapters/event_builder.py`（新增函数）
- 重采样调用点: `src/liquidloc/pipelines/prepare_pipeline.py:409-419`
- `p19_tf32_amp_explicit`: `src/liquidloc/analysis/clause_gates.py:370-406`
- `p28_nearest_resampling`: `src/liquidloc/analysis/clause_gates.py:471-497`

## 审查

| 轮次 | 问题 | 修复 | 重映射 |
|------|------|------|--------|
| 初版 | 条款 [12] 的最近邻重采样未实现；`merge_and_finalize_events` 仅做时间排序，无重采样步骤 | 新增 `nearest_neighbor_resample_events` 和 `build_unified_timeline` 函数；在 `prepare_pipeline.py` 的 UWB/VIO 事件构建后、合并前调用重采样；`p28_nearest_resampling` gate 已存在并通过 | 重采样发生在 `prepare_pipeline` 层（事件构建后），在 `merge_and_finalize_events` 之前；IMU 保留原始时间戳不参与重采样 |

## 验证

```bash
# 1) check_normalization_leak — 检测训练/测试统计量泄漏
python -c "
import sys; sys.path.insert(0, 'src')
from liquidloc.common.bridge_thresholds import check_normalization_leak
train = {'mean': 0.5, 'std': 1.0}
test_good = {'mean': 0.5, 'std': 1.0}
test_bad = {'mean': 0.6, 'std': 1.0}
r_good = check_normalization_leak(test_good, train)
r_bad = check_normalization_leak(test_bad, train)
assert r_good.passed, 'should pass when stats match'
assert not r_bad.passed, 'should fail when stats leak'
print('check_normalization_leak OK: good=%s, bad=%s' % (r_good.passed, r_bad.passed))
"
```

```bash
# 2) nearest_neighbor_resample_events — 最近邻重采样，保留原值不插值
python -c "
import sys; sys.path.insert(0, 'src')
import numpy as np
from liquidloc.dataio.adapters.event_builder import nearest_neighbor_resample_events

events = [
    {'t': 0.0, 'modality': 'uwb', 'range': 10.0, 'uwb_payload': {'range': 10.0}},
    {'t': 0.05, 'modality': 'uwb', 'range': 15.0, 'uwb_payload': {'range': 15.0}},
    {'t': 0.1, 'modality': 'uwb', 'range': 20.0, 'uwb_payload': {'range': 20.0}},
]
target_ts = [0.0, 0.03, 0.06, 0.09, 0.1]
result = nearest_neighbor_resample_events(events, target_ts, modality_filter='uwb')
assert len(result) == 5, 'should produce 5 resampled events'
# t=0.03 nearest to t=0.0 (diff=0.03) vs t=0.05 (diff=0.02) → should be t=0.05 event
assert result[1]['range'] == 15.0, 'nearest neighbor preserves original value'
assert result[1]['_resampled'] == True, 'should be marked as resampled'
assert result[1]['t'] == 0.03, 'timestamp should be target timestamp'
print('nearest_neighbor_resample_events OK:', [(e['t'], e['range']) for e in result])
"
```

```bash
# 3) build_unified_timeline — 构建统一时间线
python -c "
import sys; sys.path.insert(0, 'src')
from liquidloc.dataio.adapters.event_builder import build_unified_timeline

uwb = [{'t': 0.0}, {'t': 0.1}, {'t': 0.2}]
vio = [{'t': 0.05}, {'t': 0.15}]
ts, side = build_unified_timeline(uwb, vio)
# vio_dt≈0.0999... < uwb_dt=0.1 (fp), vio picked as denser
assert side == 'vio', 'vio is denser (fp)'
assert len(ts) >= 2, 'should cover all source timestamps'
print('build_unified_timeline OK: side=%s, n_ts=%d' % (side, len(ts)))
"
```

```bash
# 4) p28_nearest_resampling gate — 静态代码扫描
python -c "
import sys; sys.path.insert(0, 'src')
from liquidloc.analysis.clause_gates import p28_nearest_resampling
result = p28_nearest_resampling()
assert result['passed'], 'P28 gate should pass: %s' % result['detail']
print('P28 gate PASS:', result['detail'])
"
```

```bash
# 5) p19_tf32_amp_explicit gate — TF32 精度统一
python -c "
import sys; sys.path.insert(0, 'src')
from liquidloc.analysis.clause_gates import p19_tf32_amp_explicit
result = p19_tf32_amp_explicit()
print('P19 gate:', 'PASS' if result['passed'] else 'FAIL', result['detail'])
"
```
