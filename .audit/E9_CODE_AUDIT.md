# E9 实验代码审计报告

**审计范围**: configs/, src/liquidloc/, scripts/
**数据集路径约定**: configs/datasets/scene_axis_protocol.yaml, configs/datasets/sim_e9_protocol.yaml
**状态**: 审计完成

---

## 一、配置文件路由审计 (configs/)

### 1.1 configs/datasets/sim.yaml

**状态**: OK

路径约定（l18-26）:
- `raw_root: data/raw/sim_e9_10seed_40unit`
- `interim_root: data/interim/sim_e9_10seed_40unit`
- `processed_root: data/processed/sim_e9_10seed_40unit`
- `manifests_root: data/manifests/sim_e9_10seed_40unit`
- `prepare_root: outputs/prepare_sim_e9_10seed_40unit`

注释（l18）明确：异步高NLOS实验主表数据集=A2/A3 x N2/N3 四组合 x K1 x V0 x M1。
S3修复（l21）：必须扩展到 sim_e9_10seed_40unit，200条序列 x 10 seed。

**结论**: 路径约定与S3修复裁决一致，OK。

### 1.2 configs/datasets/public_dataset_registry.yaml

**状态**: 存在路径约定错误

- `sim_e9_protocol_20260726` (l108): `landed_raw_root: data/raw/sim_e9_main`
- `sim` 别名 (l143): `landed_raw_root: data/raw/sim_e9_main`

**问题**: `sim_e9_main` 不在 sim.yaml 的5层路径体系内（raw/interim/processed/manifests/prepare）。这意味着：
- sim.yaml 声明 `raw_root: data/raw/sim_e9_10seed_40unit`
- registry 指向 `data/raw/sim_e9_main`

两者不一致。HIGH-22修复注释（l119）承认"原 sim_e9_protocol_20260726 不存在"，但改成 `sim_e9_main` 仍不在任何已知路径体系内。这会导致 PreparePipeline 校验失败（prepare_pipeline.py 调用 `_enforce_sim_materialized_contract(raw_root)` 会检查目录结构）。

### 1.3 configs/datasets/scene_axis_protocol.yaml

**状态**: 缺失

该文件不存在，实际位于 `configs/base/scene_axis_protocol.yaml`。这是 **MEDIUM 风险** — 约定路径错误会导致所有引用该路径的代码在运行时找不到文件。

### 1.4 configs/datasets/sim_e9_protocol.yaml

**状态**: 完全缺失

该文件不存在。grep 搜索未发现任何相关引用。这是一个 **HIGH 风险** 的缺失文件。

### 1.5 configs/experiments/e9_dual_degradation.yaml

**状态**: 基本OK，但引用不存在的路径

关键字段:
- `experiment_id: e9_dual_degradation` (l4)
- `primary_axis: public_sequence_category` (l11)
- `dataset_name: sim` (l14)
- `seq_ids: []` (l17) — 空列表，由脚本从 frozen_axes 筛选
- `raw_root: data/raw/sim_e9_main` (l46) — 与 registry 一致，与 sim.yaml 不一致
- `available_methods_basename: [ekf, robust_ekf, lstm, liquid, transformer]` (l69) — liquid 对应 liquid_ekf

**问题**:
1. `raw_root: data/raw/sim_e9_main` 指向不存在的目录（不在 sim.yaml 体系内）
2. 注释（l6）引用 `sim_e9_protocol_20260726`，该目录不存在

### 1.7 configs/experiments/e0_traditional_only.yaml, e3_nlos.yaml

**状态**: 存在（e0, e3 已确认存在），未发现明显问题

### 1.8 configs/models/liquid_ekf.yaml

**状态**: OK

- `feature_order` 定义8个特征（dt, ax, ay, gz, range, dx, dy, dyaw）
- `window.size: 20`
- `network.output_heads: [bias, risk, uwb_scaling, vio_scaling]`
- `bridge_thresholds` 配置完整
- `train.epochs: 160`

**关键字段**:
- `project_root: E:\异步高NLOS` (l112) — Windows 路径，硬编码

### 1.9 configs/models/lstm_ekf.yaml

**状态**: 基本OK

- `feature_order` 与 liquid_ekf 一致
- `network.hidden_dim: 18`（远小于 transformer 的 64）
- `phase_schedule.warmup_epochs: 0`（无 warmup）
- `checkpoint_path: E:\异步高NLOS\outputs\lstm_full_60ep\checkpoints\...` (l85)

**问题**: `checkpoint_path` 硬编码了具体路径（lstm_full_60ep），与实际训练输出路径可能不一致。

### 1.10 configs/models/transformer_ekf.yaml

**状态**: OK

- `network.hidden_dim: 64`, `num_layers: 2`, `nhead: 4`
- `phase_schedule.warmup_epochs: 0`

---

## 二、PreparePipeline 审计 (src/liquidloc/pipelines/prepare_pipeline.py)

### 2.1 B04 强制校验

**状态**: OK

l316-320 实现 B04 强制校验：
```python
if len(seq_ids) < 20:
    raise ValueError(
        f'B04 violation: {len(seq_ids)} seq_ids provided, minimum 20 required per B04 '
        f'(test trajectory count ≥ 20 valid scorable trajectories per §9 / §14)'
    )
```

**结论**: B04 校验正确强制执行，e9_dual_degradation.yaml 的 `seq_ids: []` + frozen_axes 路由不经过 prepare_pipeline 的 `seq_ids` 参数（由脚本从 frozen_axes 筛选后传入）。

### 2.2 dataset_name 强制校验

**状态**: OK

l294: `raise ValueError('dataset_name is required')` — 强制校验 dataset_name。

### 2.3 raw_root 强制校验

**状态**: OK

l302: `raise ValueError('raw_root is required')` — 强制校验 raw_root。

**问题**: e9_dual_degradation.yaml 的 `raw_root: data/raw/sim_e9_main` 与 sim.yaml 的 `raw_root: data/raw/sim_e9_10seed_40unit` 不一致。

---

## 三、sim_materializer.py 审计 (src/liquidloc/dataio/sim_materializer.py)

### 3.1 TWR 常量

**状态**: OK

l2041-2043 定义了完整的 TWR 协议物理常量：
```python
_TWR_C = 299_702_547.0      # 光速 m/s
_TWR_REPLY_S = 1e-4          # Decawave 标准 T_reply = 100 μs
_TWR_CLOCK_JITTER_SIGMA_S = 1e-9  # tag 时钟噪声 ~1 ns
```

TWR 物理建模正确，4个时戳（twr_poll_tx, twr_poll_rx, twr_resp_tx, twr_resp_rx）正确输出。

### 3.2 锚点布局生成

**状态**: OK

l1526-1588 的 `_transform_anchor_layout` 正确：
- 读取 K 轴档位（K0/K1/K3）
- 调用 `build_anchor_layout(anchor_count, k_level, protocol_cfg["axes"]["K"], workspace_span_m=...)`
- 设置 `protocol_geometry_level`, `protocol_k_level`, `geometry_report`

### 3.3 IMU 推导

**状态**: OK

l1720-1832 的 `_derive_imu_rows_from_normalized_gt` 正确实现：
- 世界坐标系加速度转机体坐标系（l1827-1828）
- 中心差分计算加速度和角速度
- §5.1 注释明确重力处理约定（"不做任何形式的重力补偿或投影"）

### 3.4 物化流程

**状态**: 基本OK

7步注入顺序（l49-56）：
1. scene_gen
2. sensor_sim
3. async_inject
4. nlos_inject
5. missing_inject
6. resample
7. write_npz

所有步骤在 sim_materializer.py 中实现。

### 3.5 数据集路径约定

**状态**: 未找到硬编码

sim_materializer.py 中未发现对 `sim_e9_10seed_40unit` 或 `sim_e9_main` 的硬编码引用。数据集路径由调用方传入 `raw_root` 参数。

---

## 四、EKF 核心审计 (src/liquidloc/estimators/ekf_core.py)

### 4.1 VIO 参考位姿过时检查

**状态**: OK

l1128-1133 实现 VIO 参考位姿时效性检查：
```python
ref_pose_stale = (
    self._last_vio_reference_pose is not None
    and self._last_vio_reference_pose_timestamp is not None
    and self._timestamp is not None
    and (self._timestamp - self._last_vio_reference_pose_timestamp) > VIO_REF_POSE_STALE_SECONDS
)
```

阈值 `VIO_REF_POSE_STALE_SECONDS` 从 `liquidloc/common/constants.py` 导入。

### 4.2 UWB 创新协方差抖动

**状态**: OK

l926-943 实现 UWB 路径的协方差抖动：
```python
if S <= 0.0:
    cov_jitter_eps = float(BRIDGE_THRESHOLDS["cov_jitter_eps"])
    jittered_S = S + cov_jitter_eps
    if jittered_S > 0.0:
        S = jittered_S
    else:
        return {...}  # fail-loud
```

### 4.3 IMU 缺失膨胀

**状态**: OK

l756-767 实现 IMU 数据缺失时的过程噪声膨胀：
```python
if isinstance(imu_missing_mask, (list, tuple)) and len(imu_missing_mask) >= 3:
    imu_field_missing = [bool(m) for m in imu_missing_mask[:3]]
    if any(imu_field_missing):
        imu_missing_inflation = float(BRIDGE_THRESHOLDS["imu_missing_inflation"])
```

---

## 五、协议层审计 (src/liquidloc/protocol/)

### 5.1 bridge_thresholds.py

**状态**: OK

`BRIDGE_THRESHOLDS` 提供所有桥接层硬阈值的单源真相，包括：
- `imu_missing_inflation: 10.0`（IMU 缺失膨胀10倍）
- `cov_jitter_eps: 1e-8`（协方差抖动 epsilon）
- `vio_hard_skip_quality_floor: 0.01`
- `uwb_hard_skip_quality_floor: 0.01`

### 5.2 scene_axis_protocol.py

**状态**: OK

`load_scene_axis_protocol()` 正确加载 `configs/base/scene_axis_protocol.yaml`。

---

## 六、脚本审计 (scripts/)

### 6.1 脚本中硬编码路径

**问题**: 多个脚本硬编码 `sim_e9_protocol_20260726` 路径：

- `scripts/02_generate_sim_raw.py`: `default=str(ROOT / "data" / "raw" / "sim_e9_protocol_20260726")`
- `scripts/07_run_baselines.py`: `_SIM_DEFAULT_PREPARE_ROOT = Path("outputs/prepare_sim_e9_protocol_20260726")`
- `scripts/06_train_liquid.py`, `scripts/05_train_lstm.py`, `scripts/07_train_transformer.py`: 类似硬编码

这些硬编码与 sim.yaml 的 `prepare_root: outputs/prepare_sim_e9_10seed_40unit` 不一致。

---

## 七、问题汇总

### HIGH 严重问题

| ID | 文件 | 行号 | 问题 | 影响 |
|----|------|------|------|------|
| H1 | configs/datasets/sim_e9_protocol.yaml | - | **文件不存在** | 所有引用该路径的代码会失败 |
| H2 | configs/datasets/public_dataset_registry.yaml | 120, 149 | `landed_raw_root: data/raw/sim_e9_main` 不在 sim.yaml 体系内 | PreparePipeline 校验失败 |
| H3 | configs/experiments/e9_dual_degradation.yaml | 46 | `raw_root: data/raw/sim_e9_main` 与 sim.yaml 不一致 | 数据加载路径错误 |

### MEDIUM 中等问题

| ID | 文件 | 行号 | 问题 | 影响 |
|----|------|------|------|------|
| M1 | configs/datasets/scene_axis_protocol.yaml | - | **文件不存在于约定路径**（实际在 configs/base/） | 引用约定路径的代码会失败 |
| M3 | scripts/*.py | 多处 | 硬编码 `sim_e9_protocol_20260726` 路径 | 与 sim.yaml 声明不一致 |

### LOW 轻微问题

| ID | 文件 | 行号 | 问题 | 影响 |
|----|------|------|------|------|
| L1 | configs/models/lstm_ekf.yaml | 85 | `checkpoint_path` 硬编码 `lstm_full_60ep` | 可能与实际训练输出不一致 |
| L2 | configs/models/liquid_ekf.yaml, lstm_ekf.yaml, transformer_ekf.yaml | 112, 86, 88 | `project_root: E:\异步高NLOS` 硬编码 Windows 路径 | 跨平台可移植性问题 |

---

## 八、修复建议

### H1 修复方案
创建 `configs/datasets/sim_e9_protocol.yaml`，内容应与 `configs/base/scene_axis_protocol.yaml` 的 sim 数据集部分保持一致。

### H2-H3 修复方案
统一数据路径约定。选择以下方案之一：

**方案A**: 将 sim.yaml 的 raw_root 改为 `data/raw/sim_e9_main`（推荐，因为实际已物化）
```yaml
raw_root: data/raw/sim_e9_main
interim_root: data/interim/sim_e9_main
processed_root: data/processed/sim_e9_main
manifests_root: data/manifests/sim_e9_main
prepare_root: outputs/prepare_sim_e9_main
```

**方案B**: 将 registry 的 landed_raw_root 改为 `data/raw/sim_e9_10seed_40unit`
```yaml
landed_raw_root: data/raw/sim_e9_10seed_40unit
```

### M1 修复方案
更新所有引用 `configs/datasets/scene_axis_protocol.yaml` 的代码，改为 `configs/base/scene_axis_protocol.yaml`，或创建符号链接。

### M3 修复方案
将脚本中的硬编码路径改为从 sim.yaml 动态读取，或使用命令行参数传入。

---

## 九、审计结论

E9 实验代码整体架构清晰，协议层设计完善（BRIDGE_THRESHOLDS 单源真相、B04 强制校验、TWR 物理建模正确）。但存在以下关键问题：

1. **路径约定不一致**: sim.yaml、public_dataset_registry、e9_dual_degradation.yaml 三处 raw_root 相互矛盾
2. **缺失文件**: sim_e9_protocol.yaml 不存在，scene_axis_protocol.yaml 路径错误

建议优先修复 H1-H5（所有 HIGH 问题），然后处理 M1-M3（MEDIUM 问题）。
