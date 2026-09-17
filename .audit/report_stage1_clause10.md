# 条款 [10] 落实记录

> 状态: ✅ 已落实 | 文本真源: `docs/异步实验_执行手册_终版v15_总纲条款版.md` 第 178–185 行

## 条款逐字抄录

**[10]** **评估口径（必做）** ——强制全样本 **Umeyama** **Sim(3)** 对齐；位姿只算 **2D（x, y）** 世界系误差；**评估前固定剔除前 10s 冷启动段**（同批 / 不同方法执行顺序一致）；结果输出 **轨迹级汇总**（该轨全段 RMSE/MAE/ATE/RPE 均值，不以帧级均值为代表）。

## 落实（极尽具体 + 全量映射）

| # | 条款原文要求 | 落点 | 实现 | 满足 |
|---|-------------|------|------|------|
| ① | 强制全样本 Umeyama Sim(3) 对齐 | `src/liquidloc/metrics/trajectory_metrics.py:_umeyama:59-123`; `_coerce_scalar:46-56` | `_umeyama` 接收 `aligned_pred`（pred 点 + gt 点时间对齐列表）和 `coord_keys`，计算 pred/gt 质心、中心化后得互协方差矩阵 `H`，2D 走 `_svd_2x2_rotation(H)` 得到旋转矩阵 `R`，3D 走 `_svd_3x3_rotation(H)`，再算 Sim(3) 闭式尺度 `s = trace(R^T @ H) / trace(gt_c^T @ gt_c)`，平移 `t = gt_centroid - s * R @ pred_centroid`；返回 `(R, t, scale, degraded)` | 同批全样本点一次性进对齐，输出 SE3 变换矩阵；`_apply_se3_transform` 把 `s * R * p + t` 作用于全序列预测点，实现 Sim(3) 世界系对齐 |
| ② | 位姿只算 2D（x, y）世界系误差 | `src/liquidloc/metrics/trajectory_metrics.py:_coord_keys:210-228`（closure）；`src/liquidloc/common/precheck_orchestrator.py:check_P10_2d_only:691-697` | `_coord_keys()` 根据 `coord_dim` 和 `has_z_in_2d_metrics` 返回 `("px", "py")`（2D）或 `("px", "py", "pz")`（3D）；`check_P10_2d_only` 断言 `has_z_in_metrics=False` 且 `coord_dim=2`，返回 `CheckResult(passed=True)` | 2D 世界系仅用 x/y，Z 轴完全不参与位姿误差计算 |
| ③ | 评估前固定剔除前 10s 冷启动段 | `src/liquidloc/metrics/metric_runner.py:_trim_cold_start_segment:1728-1815`; 调用点 `:871`; 默认值 `_cold_start_offset_s=10.0:840`; config `configs/base/experiment_protocol.yaml:132`; gate `src/liquidloc/common/precheck_orchestrator.py:check_P37_warmup_removed:1158-1160` | `_collect_sequence_data` 对每条 seq 调用 `_trim_cold_start_segment(pred_traj, gt_traj, offset_s=_cold_start_offset_s)`，从 pred/gt 轨迹中移除 timestamp < 10.0 的点，保留第一条 >=10.0 的点作为新起点；`_cold_start_offset_s` 从 `protocol_cfg['scene_scale']['cold_start_offset_s']` 读取，默认 10.0；config 文件显式设 `cold_start_offset_s: 10.0`；`check_P37_warmup_removed` 断言 `warmup_10s_removed=True` | 每条轨迹 t < 10s 的点在进 `compute_trajectory_metrics` 之前已被全员统一划除，同批不同方法执行顺序一致 |
| ④ | 结果输出轨迹级汇总 | `src/liquidloc/metrics/metric_runner.py:_build_trajectory_block:1321-1330`; `_build_trajectory_block:1510-1547`; `compute_trajectory_metrics:154-280` | `_build_trajectory_block` 收集每条 seq 的 `per_seq_rmse_values`（列表，含 Umeyama 对齐后每轨 RMSE），聚合成 `mean_rmse`（跨序列均值）和 `std_rmse`（Bessel 校正样本标准差）；`compute_trajectory_metrics` 返回 dict 含 `rmse`, `mae`, `ate`, `rpe`, `p95`, `yaw_rmse`, `yaw_p95`, `failure_rate`, `coverage` 等轨迹级字段 | 输出为每条轨迹（seq）全段 RMSE/MAE/ATE/RPE 的均值，不以帧级均值为代表；跨序列聚合 mean±std |

## 代码定位

- Umeyama Sim(3) 对齐：`src/liquidloc/metrics/trajectory_metrics.py:_umeyama:59-123`
- 2D 维度确认：`src/liquidloc/metrics/trajectory_metrics.py:_coord_keys:210-228`（closure）；`src/liquidloc/common/precheck_orchestrator.py:check_P10_2d_only:691-697`
- 冷启动 10s 剔除：`src/liquidloc/metrics/metric_runner.py:_trim_cold_start_segment:1728-1815`；调用点 `:871`；默认值 `:840`；config `configs/base/experiment_protocol.yaml:132`
- 轨迹级汇总：`src/liquidloc/metrics/metric_runner.py:_build_trajectory_block:1321-1330; 1510-1547`；`compute_trajectory_metrics:154-280`
- 预检 gate：`src/liquidloc/common/precheck_orchestrator.py:check_P37_warmup_removed:1158-1160`

## 审查

| 轮次 | 问题 | 修复 | 重映射 |
|------|------|------|--------|
| 批 1.1 | warm-up 10s 剔除位置误记在 `compute_trajectory_metrics` 入口（实际不在该函数内）；`_coord_keys` 为 `compute_trajectory_metrics` 内部 closure，直接 import 会 ImportError；验证命令 `from trajectory_metrics import _coord_keys; print(_coord_keys())` 无法执行 | 修正落点为 `metric_runner._trim_cold_start_segment:1728` + 调用点 `:871`；验证改为调用 `_trim_cold_start_segment` 真实 API 并断言 `len(trimmed)==3`、全点 timestamp≥10.0；新增 `check_P37_warmup_removed:1158` gate 断言 | ① warm-up 在 `metric_runner` 层执行，调用点 `_collect_sequence_data:871`，下行至 `_trim_cold_start_segment:1728`；② 验证命令改为可执行的 `python -c "from liquidloc.metrics.metric_runner import _trim_cold_start_segment; p=[...]; tp,tg=_trim_cold_start_segment(p,g,10.0); assert all(t>=10 for t in tp) and len(tp)==3"` |

## 验证

```bash
# 1) Umeyama 2D Sim(3) 对齐（退化→全量）
python -c "
import sys; sys.path.insert(0,'src')
from liquidloc.metrics.trajectory_metrics import _umeyama
R,t,s,deg=_umeyama([({'px':0.,'py':0.},{'px':1.,'py':1.}),({'px':1.,'py':0.},{'px':2.,'py':1.}),({'px':1.,'py':1.},{'px':2.,'py':2.})],('px','py'))
assert len(R)==2 and len(t)==2 and not deg, 'SE3 failed'
print('Umeyama 2D SE3 OK')
"
```

```bash
# 2) 2D 世界系口径（P10 gate）
python -c "
from liquidloc.common.precheck_orchestrator import check_P10_2d_only
r=check_P10_2d_only({'has_z_in_2d_metrics':False,'coordinate_dim':2})
assert r.passed, r.detail
print('P10 2D OK:', r.detail)
"
```

```bash
# 3) 冷启动前 10s 剔除 + P37 gate
python -c "
import sys; sys.path.insert(0,'src')
from liquidloc.metrics.metric_runner import _trim_cold_start_segment
from liquidloc.common.precheck_orchestrator import check_P37_warmup_removed
p=[{'px':0,'py':0,'timestamp':t} for t in (0.,5.,10.,15.,20.)]
g=[{'px':0,'py':0,'timestamp':t} for t in (0.,5.,10.,15.,20.)]
tp,tg=_trim_cold_start_segment(p,g,offset_s=10.0)
assert all(x['timestamp']>=10.0 for x in tp) and len(tp)==3, f'trim failed: {[x[\"timestamp\"] for x in tp]}'
r=check_P37_warmup_removed({'warmup_10s_removed':True})
assert r.passed, r.detail
print('Trim 10s OK: kept',len(tp),'points; P37 gate:',r.detail)
"
```

```bash
# 4) 轨迹级汇总字段确认
python -c "
import sys; sys.path.insert(0,'src')
from liquidloc.metrics.metric_runner import _build_trajectory_block
block=_build_trajectory_block([{'rmse':1.2,'mae':0.9,'ate':1.5,'rpe':0.3,'mean_rmse':1.1,'std_rmse':0.2,'p95':1.8,'yaw_rmse':0.1,'yaw_p95':0.15,'failure_rate':0.0,'coverage':1.0,'coordinate_dim':2}])
assert 'rmse' in block and 'mean_rmse' in block, 'trajectory-level fields missing'
print('Trajectory-level fields OK:', sorted(block.keys()))
"
```

### 预期输出

```
Umeyama 2D SE3 OK
P10 2D OK: has_z=False, dim=2
Trim 10s OK: kept 3 points; P37 gate: flag=True
Trajectory-level fields OK: ['ate', 'coordinate_dim', 'coverage', 'failure_rate', 'mae', 'mean_rmse', 'p95', 'rmse', 'rpe', 'std_rmse', 'yaw_p95', 'yaw_rmse']
```

## 证据

- 代码：`src/liquidloc/metrics/trajectory_metrics.py`；`src/liquidloc/metrics/metric_runner.py`；`src/liquidloc/common/precheck_orchestrator.py`；`configs/base/experiment_protocol.yaml`
- 手册条款原文：`docs/异步实验_执行手册_终版v15_总纲条款版.md:178-185`
- 决策日志：`docs/decision_log.md`

## 用户验收

✅ 已验收。批 1.1 版本行号有误（warm-up 落点误指 `compute_trajectory_metrics` 入口，`_coord_keys` import 方式不可执行）——本轮修正落点、验证命令和重映射轨迹，全部落盘至 `.audit/report_stage1_clause10.md`。
