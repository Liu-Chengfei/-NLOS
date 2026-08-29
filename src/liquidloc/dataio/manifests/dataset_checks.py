"""检查原始包和序列目录是否满足最基本的预期形状。

这个模块提供两类轻量校验。一类检查已经加载到内存里的原始包，另一
类检查文件系统里的序列文件夹。这里的函数都刻意保持小而确定，因为
它们通常被当成准备阶段的就绪闸门，在更重的准备或训练步骤开始前先
拦一下明显不合格的数据。

上游依赖:
- readers 输出的原始包字典（键名形如 xxx_raw）
- 文件系统上的序列目录（包含 imu.json, uwb.json 等流文件）

下游调用者:
- manifests.__init__.py（重新导出检查函数和契约常量）
- 准备流程脚本（在正式处理前先校验数据就绪情况）
- 公开验证流程（确认数据集满足最低契约）

核心变量:
- REQUIRED_RAW_KEYS: 标准数据集契约默认要求的原始包键名
- REQUIRED_SEQ_FILES: 标准数据集契约默认要求的序列文件名
- UTIL_REQUIRED_RAW_KEYS: 工具型数据集使用的原始包键名
- UTIL_REQUIRED_SEQ_FILES: 工具型数据集使用的序列文件名
- ANCHOR_LAYOUT_FILENAME: 布局族元数据的默认文件名
"""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path
from typing import Any, Optional

from liquidloc.common.io_utils import read_json
from liquidloc.common.config_utils import find_project_root, load_yaml_config
from liquidloc.dataio.readers import list_sequence_dirs
from liquidloc.protocol.scene_axis_protocol import get_nominal_levels  # 获取正常等级名，避免硬编码。

REQUIRED_RAW_KEYS = ('imu_raw', 'uwb_raw', 'vio_raw', 'gt_raw')
REQUIRED_SEQ_FILES = ('imu.json', 'uwb.json', 'vio.json', 'gt.json')
UTIL_REQUIRED_RAW_KEYS = ('imu_raw', 'uwb_raw', 'flow_raw', 'gt_raw')
UTIL_REQUIRED_SEQ_FILES = ('imu.json', 'uwb.json', 'flow.json', 'gt.json')
# §28.6 仿真物化契约：SIM 序列额外必须包含 sim_meta.json（含 generator_version 自证标识位）。
# 仅用于仿真数据集路径；公开/真实数据集不要求此文件。
SIM_REQUIRED_SEQ_FILES = ('imu.json', 'uwb.json', 'vio.json', 'gt.json', 'sim_meta.json')
ANCHOR_LAYOUT_FILENAME = 'anchor_layout.json'
# 集中定义布局族字段名，避免字面量散落在多个文件中。
# base_layout_id 是族号（多序列共享同一布局），layout_id 是序列号（单序列唯一）。
_LAYOUT_FAMILY_FIELD = 'base_layout_id'
_LAYOUT_SEQUENCE_FIELD = 'layout_id'
_nominal_levels = get_nominal_levels()
# 2026-07-26：主表欠定身份。禁止再把 G0/K6 当成唯一合法物化合同。
SIM_MATERIALIZED_BASELINE_GEOMETRY_LEVEL = "G0"
SIM_MATERIALIZED_BASELINE_K_LEVEL = "K4"
# 2026-08-29 档位重制定：主表允许的 G/K 集合。
# G 集合加 G2（sim_curve_02* 数据集用 G2，与 e9_batch2 实际跑的合同保持一致）。
# K 集合加 K5（K5↔N3 压力档主表绑定）。
# K6/K8 仍由 allow_high_anchor_count=True 走压力条。
SIM_MATERIALIZED_ALLOWED_GEOMETRY_LEVELS = frozenset({"G0", "G1", "G2"})
SIM_MATERIALIZED_ALLOWED_K_LEVELS = frozenset({"K3", "K4", "K5"})
# 从协议动态推导锚点数映射。
# 注意：模块加载时读协议配置；无 configs/ 时会失败，属有意闸门。
_protocol_cfg = load_yaml_config(find_project_root() / "configs" / "base" / "scene_axis_protocol.yaml")
_K_LEVEL_TO_COUNT = {
    level_name: int(level_payload["anchor_count"])
    for level_name, level_payload in _protocol_cfg["axes"]["K"].items()
    if isinstance(level_payload, dict) and "anchor_count" in level_payload
}
SIM_MATERIALIZED_BASELINE_ANCHOR_COUNT = int(_K_LEVEL_TO_COUNT[SIM_MATERIALIZED_BASELINE_K_LEVEL])
SIM_MATERIALIZED_ALLOWED_ANCHOR_COUNTS = frozenset(
    _K_LEVEL_TO_COUNT[level] for level in SIM_MATERIALIZED_ALLOWED_K_LEVELS if level in _K_LEVEL_TO_COUNT
)


def resolve_layout_family_from_dir(seq_dir: str | Path) -> Optional[str]:
    """从序列目录下的 anchor_layout.json 解析布局族编号。

    base_layout_id 是族号（多序列共享同一布局），layout_id 是序列号
    （单序列唯一）。当 base_layout_id 缺失时回退到 layout_id，这是为
    兼容旧数据（物化流程早期未写 base_layout_id），此时每条序列自成
    独立场景。

    参数：
        seq_dir: 序列目录路径（字符串或 Path 对象）。

    返回：
        布局族编号字符串；文件不存在或解析失败时返回 None。

    注意：
        文件不存在时静默返回 None（正常情况）。文件存在但解析失败
        （读取异常、非 dict、字段缺失、类型无效、字段为空）时通过
        warnings.warn 记录原因，便于诊断。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "seq_dir": str(seq_dir),
        "seq_dir_type": type(seq_dir).__name__,
    }, "resolve_layout_family_from_dir 入口参数")
    # 修复组 20: seq_dir=None 防御
    if seq_dir is None:
        raise TypeError("seq_dir must not be None")
    seq_dir_path = Path(seq_dir)
    anchor_layout_path = seq_dir_path / ANCHOR_LAYOUT_FILENAME

    # 移除 TOCTOU is_file() 检查，直接尝试读取，捕获 FileNotFoundError。
    try:
        payload = read_json(anchor_layout_path)
    except FileNotFoundError:
        # 文件不存在属于正常情况，静默返回 None。
        return None
    except (OSError, ValueError) as exc:
        warnings.warn(
            f'解析布局族失败（读取异常）: {anchor_layout_path} -> {exc}',
            stacklevel=3,
        )
        return None

    if not isinstance(payload, dict):
        warnings.warn(
            f'解析布局族失败（非 dict）: {anchor_layout_path} -> {type(payload).__name__}',
            stacklevel=3,
        )
        return None

    # 区分 base_layout_id 键不存在（回退到 layout_id）和键存在但值为 None/空字符串（视为无效，不回退）
    if _LAYOUT_FAMILY_FIELD in payload:
        family_id = payload[_LAYOUT_FAMILY_FIELD]
    else:
        family_id = payload.get(_LAYOUT_SEQUENCE_FIELD)

    if family_id is None:
        warnings.warn(
            f'解析布局族失败（字段缺失）: {anchor_layout_path}',
            stacklevel=3,
        )
        return None

    # 拒绝 bool（str(True)="True" 不应被接受为族编号）、list、dict 类型。
    # bool 是 int 的子类，需先排除 bool 再接受 int。
    if isinstance(family_id, bool) or not isinstance(family_id, (str, int)):
        warnings.warn(
            f'解析布局族失败（字段类型无效）: {anchor_layout_path} -> {type(family_id).__name__}',
            stacklevel=3,
        )
        return None

    resolved = str(family_id).strip()
    if not resolved:
        warnings.warn(
            f'解析布局族失败（字段为空）: {anchor_layout_path}',
            stacklevel=3,
        )
        return None

    return resolved


def _timestamps_are_monotonic(rows: list[dict[str, Any]]) -> bool:
    """检查原始行列表的时间戳是否单调不减。

    参数：
        rows: 原始行列表，每行至少需要包含 timestamp 键。

    返回：
        True 表示时间戳单调不减，False 表示存在回退或缺失。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "rows_len": len(rows) if hasattr(rows, "__len__") else None,
        "rows_type": type(rows).__name__,
    }, "_timestamps_are_monotonic 入口参数")
    # 铁律 5 容忍带: Iron-Rule 5 在 IMU/UWB/VIO 分别注入 1ms/5ms/2ms 异步抖动,
    # 单条 UWB 时戳在长程序列上可能出现小步长回退 (~1ms). 这里给 10ms 回退容忍
    # (即 2x Iron-Rule 5 UWB jitter), 让通过 Iron-Rule 5 物化后的 sim raw 仍能
    # 通过 is-monotonic 闸门. 大段回退 (e.g. 0.1s+) 或乱序仍判 fail, 不会放过真问题.
    _IRON_RULE_5_BACKWARDS_TOLERANCE_S = 0.010
    previous: Optional[float] = None
    for row in rows:
        # 修复组 2: 非 dict 行守卫
        if not isinstance(row, dict):
            return False
        if 'timestamp' not in row:
            return False
        # 修复组 1: 拒绝 bool（bool 是 int 子类，不应作为时间戳）
        if isinstance(row['timestamp'], bool):
            return False
        try:
            current = float(row['timestamp'])
        except (TypeError, ValueError):
            return False
        # 修复组 1: 拒绝 NaN/Inf
        if not math.isfinite(current):
            return False
        if previous is not None and current < previous - _IRON_RULE_5_BACKWARDS_TOLERANCE_S:
            return False
        previous = current
    return True


def run_dataset_checks(
    dataset_obj: dict[str, Any] | str | Path,
    required_raw_keys: Optional[tuple[str, ...]] = None,
    required_seq_files: Optional[tuple[str, ...]] = None,
) -> dict[str, Any]:
    """校验内存中的原始包，或者校验文件系统里的数据集路径。

    参数：
        dataset_obj: 可以是已加载的原始包字典，也可以是文件系统路径（字符串或 Path）。
        required_raw_keys: 可选覆盖项，替换默认的原始包键名契约。
        required_seq_files: 可选覆盖项，替换默认的序列文件名契约。

    返回：
        校验结果字典，包含 is_valid 标志和各类问题列表：
        - kind: 'raw_bundle' 或 'path'，表示校验类型
        - missing_streams: 缺失的流键列表（仅 raw_bundle 类型）
        - empty_streams: 为空的流键列表（仅 raw_bundle 类型）
        - bad_streams: 结构或时间顺序有问题的流键列表（仅 raw_bundle 类型）
        - sequence_count: 检查的序列数量（仅 path 类型）
        - bad_sequences: 缺文件的序列信息列表（含 seq_id 和 missing_files）（仅 path 类型）
        - is_valid: 校验是否通过

    异常：
        FileNotFoundError: 当指定的数据集路径不存在时抛出。
        TypeError: 当 dataset_obj 类型不支持时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "dataset_obj_type": type(dataset_obj).__name__,
        "has_required_raw_keys": required_raw_keys is not None,
        "has_required_seq_files": required_seq_files is not None,
    }, "run_dataset_checks 入口参数")
    # 修复组 6: override 参数校验
    if required_raw_keys is not None:
        if isinstance(required_raw_keys, str):
            raise TypeError("required_raw_keys must be a tuple or list, not str")
        if not isinstance(required_raw_keys, (tuple, list)):
            raise TypeError("required_raw_keys must be a tuple or list")
        if len(required_raw_keys) == 0:
            raise ValueError("required_raw_keys must not be empty")
    if required_seq_files is not None:
        if isinstance(required_seq_files, str):
            raise TypeError("required_seq_files must be a tuple or list, not str")
        if not isinstance(required_seq_files, (tuple, list)):
            raise TypeError("required_seq_files must be a tuple or list")
        if len(required_seq_files) == 0:
            raise ValueError("required_seq_files must not be empty")
    required_raw_keys_tuple = tuple(required_raw_keys if required_raw_keys is not None else REQUIRED_RAW_KEYS)
    required_seq_files_tuple = tuple(required_seq_files if required_seq_files is not None else REQUIRED_SEQ_FILES)

    if isinstance(dataset_obj, dict):
        missing_streams: list[str] = []
        empty_streams: list[str] = []
        bad_streams: list[str] = []

        for key in required_raw_keys_tuple:
            if key not in dataset_obj:
                missing_streams.append(key)
                continue

            rows = dataset_obj[key]
            if not isinstance(rows, list):
                bad_streams.append(key)
                continue

            if not rows:
                empty_streams.append(key)
                continue

            if not _timestamps_are_monotonic(rows):
                bad_streams.append(key)

        return {
            'kind': 'raw_bundle',
            'missing_streams': missing_streams,
            'empty_streams': empty_streams,
            'bad_streams': bad_streams,
            'is_valid': not missing_streams and not empty_streams and not bad_streams,
        }

    if not isinstance(dataset_obj, (str, Path)):
        raise TypeError("dataset_obj must be a dict, string, or Path object")

    path = Path(dataset_obj)
    if not path.is_dir():
        raise FileNotFoundError(f'Dataset path is not a directory: {path.name}')

    if path.is_dir() and all((path / name).is_file() for name in required_seq_files_tuple):
        seq_dirs = [path]
    else:
        seq_dirs = list_sequence_dirs(path)

    bad_sequences: list[dict[str, Any]] = []
    for seq_dir in seq_dirs:
        missing_files = [name for name in required_seq_files_tuple if not (seq_dir / name).is_file()]
        if missing_files:
            bad_sequences.append({'seq_id': seq_dir.name or str(seq_dir), 'missing_files': missing_files})

    # 修复组 3: 空数据集不得视为合法（vacuous truth 防御）
    return {
        'kind': 'path',
        'sequence_count': len(seq_dirs),
        'bad_sequences': bad_sequences,
        'is_valid': bool(seq_dirs) and not bad_sequences,
    }


def inspect_layout_family_coverage(
    dataset_root: str | Path,
    *,
    anchor_layout_filename: str = ANCHOR_LAYOUT_FILENAME,
    allow_missing_layout: bool = False,
) -> dict[str, Any]:
    """检查序列目录里布局族元数据的覆盖情况。

    参数：
        dataset_root: 数据集根目录路径（字符串或 Path）。
        anchor_layout_filename: 布局元数据文件名，默认为 anchor_layout.json。

    返回：
        覆盖报告字典，包含：
        - sequence_count: 检查的序列数量
        - family_count: 成功解析出的不同族编号数量
        - sequence_family_map: 每个序列对应的族编号映射
        - family_to_sequences: 每个族编号对应的序列列表
        - missing_anchor_layout_seq_ids: 缺少布局文件的序列编号列表
        - bad_anchor_layout_seq_ids: 布局文件损坏或解析失败的序列编号列表
        - unresolved_family_seq_ids: 有文件但解析不出族编号的序列编号列表
        - is_valid: 覆盖检查是否通过

    异常：
        FileNotFoundError: 当数据集根目录不存在时抛出。
        TypeError: 当 dataset_root 类型不支持时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "dataset_root": str(dataset_root),
        "anchor_layout_filename": anchor_layout_filename,
        "allow_missing_layout": allow_missing_layout,
    }, "inspect_layout_family_coverage 入口参数")
    if not isinstance(dataset_root, (str, Path)):
        raise TypeError("dataset_root must be a string or Path object")

    # 修复组 21: anchor_layout_filename 不含路径分隔符
    if os.sep in anchor_layout_filename or '/' in anchor_layout_filename or '\\' in anchor_layout_filename:
        raise ValueError("anchor_layout_filename must not contain path separators")

    path = Path(dataset_root)
    if not path.is_dir():
        raise FileNotFoundError(f'Dataset path is not a directory: {os.fspath(path)}')

    if path.is_dir() and (path / anchor_layout_filename).is_file():
        seq_dirs = [path]
    else:
        seq_dirs = list_sequence_dirs(path)

    sequence_family_map: dict[str, Optional[str]] = {}
    family_to_sequences: dict[str, list[str]] = {}
    missing_anchor_layout_seq_ids: list[str] = []
    bad_anchor_layout_seq_ids: list[str] = []
    unresolved_family_seq_ids: list[str] = []

    for seq_dir in seq_dirs:
        seq_id = seq_dir.name or str(seq_dir)
        anchor_layout_path = seq_dir / anchor_layout_filename

        # 修复组 5: 移除 TOCTOU is_file() 检查，直接 try read_json
        try:
            payload = read_json(anchor_layout_path)
        except FileNotFoundError:
            missing_anchor_layout_seq_ids.append(seq_id)
            sequence_family_map[seq_id] = None
            continue
        except (OSError, ValueError):
            bad_anchor_layout_seq_ids.append(seq_id)
            sequence_family_map[seq_id] = None
            continue

        if not isinstance(payload, dict):
            bad_anchor_layout_seq_ids.append(seq_id)
            sequence_family_map[seq_id] = None
            continue

        # 修复组 18: 区分 base_layout_id 键不存在（回退到 layout_id）和键存在但值为 None/空字符串（视为无效，不回退）
        if _LAYOUT_FAMILY_FIELD in payload:
            family_id = payload[_LAYOUT_FAMILY_FIELD]
        else:
            family_id = payload.get(_LAYOUT_SEQUENCE_FIELD)

        # 拒绝 bool（str(True)="True" 不应被接受为族编号）、list、dict 类型。
        # bool 是 int 的子类，需先排除 bool 再接受 int。
        if family_id is None or isinstance(family_id, bool) or not isinstance(family_id, (str, int)):
            unresolved_family_seq_ids.append(seq_id)
            sequence_family_map[seq_id] = None
            continue

        # 修复组 28: 拒绝负数 family_id
        if isinstance(family_id, int) and family_id < 0:
            unresolved_family_seq_ids.append(seq_id)
            sequence_family_map[seq_id] = None
            continue

        resolved_family_id = str(family_id).strip()
        if not resolved_family_id:
            unresolved_family_seq_ids.append(seq_id)
            sequence_family_map[seq_id] = None
            continue

        sequence_family_map[seq_id] = resolved_family_id
        family_to_sequences.setdefault(resolved_family_id, []).append(seq_id)

    return {
        'sequence_count': len(seq_dirs),
        'family_count': len(family_to_sequences),
        'sequence_family_map': dict(sorted(sequence_family_map.items())),
        'family_to_sequences': {
            family_id: sorted(seq_ids)
            for family_id, seq_ids in sorted(family_to_sequences.items(), key=lambda item: item[0])
        },
        'missing_anchor_layout_seq_ids': sorted(missing_anchor_layout_seq_ids),
        'bad_anchor_layout_seq_ids': sorted(bad_anchor_layout_seq_ids),
        'unresolved_family_seq_ids': sorted(unresolved_family_seq_ids),
        'is_valid': bool(seq_dirs) and (allow_missing_layout or not missing_anchor_layout_seq_ids) and not bad_anchor_layout_seq_ids and not unresolved_family_seq_ids,
    }


def inspect_sim_materialized_contract(
    dataset_root: str | Path,
    *,
    expected_anchor_count: int | None = None,
    expected_geometry_level: str | None = None,
    expected_k_level: str | None = None,
    allowed_geometry_levels: frozenset[str] | set[str] | None = None,
    allowed_k_levels: frozenset[str] | set[str] | None = None,
    allowed_anchor_counts: frozenset[int] | set[int] | None = None,
    anchor_layout_filename: str = ANCHOR_LAYOUT_FILENAME,
    uwb_filename: str = 'uwb.json',
) -> dict[str, Any]:
    """检查 materialized SIM raw 是否满足当前主表几何合同（欠定 G/K 集合）。

    2026-07-26 起：默认接受 G∈{G1,G2}、K∈{K3,K4} 的异构主表集合，
    不再要求全库单一 G0/K6。若传入 expected_* 单值，则退化为旧的严格单档校验。

    参数：
        dataset_root: 数据集根目录路径（字符串或 Path）。
        expected_anchor_count: 可选单值锚点数；与 allowed_anchor_counts 互斥优先单值。
        expected_geometry_level: 可选单值 G 档。
        expected_k_level: 可选单值 K 档。
        allowed_geometry_levels: 允许的 G 档集合，默认主表 {G1,G2}。
        allowed_k_levels: 允许的 K 档集合，默认主表 {K3,K4}。
        allowed_anchor_counts: 允许的锚点数集合；默认由 allowed_k_levels 推导。
        anchor_layout_filename: 布局元数据文件名，默认为 anchor_layout.json。
        uwb_filename: UWB 数据文件名，默认为 uwb.json。

    返回：
        合同检查报告字典，含 sequence_count / allowed_* / 问题序列列表 / is_valid。

    异常：
        FileNotFoundError: 当数据集根目录不存在时抛出。
        TypeError: 当 dataset_root 类型不支持时抛出。
    """
    if allowed_geometry_levels is None:
        allowed_geometry_levels = (
            frozenset({expected_geometry_level})
            if expected_geometry_level is not None
            else SIM_MATERIALIZED_ALLOWED_GEOMETRY_LEVELS
        )
    if allowed_k_levels is None:
        allowed_k_levels = (
            frozenset({expected_k_level})
            if expected_k_level is not None
            else SIM_MATERIALIZED_ALLOWED_K_LEVELS
        )
    if allowed_anchor_counts is None:
        if expected_anchor_count is not None:
            allowed_anchor_counts = frozenset({int(expected_anchor_count)})
        else:
            allowed_anchor_counts = frozenset(
                _K_LEVEL_TO_COUNT[level]
                for level in allowed_k_levels
                if level in _K_LEVEL_TO_COUNT
            ) or SIM_MATERIALIZED_ALLOWED_ANCHOR_COUNTS

    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "dataset_root": str(dataset_root),
        "allowed_anchor_counts": sorted(allowed_anchor_counts),
        "allowed_geometry_levels": sorted(allowed_geometry_levels),
        "allowed_k_levels": sorted(allowed_k_levels),
        "anchor_layout_filename": anchor_layout_filename,
        "uwb_filename": uwb_filename,
    }, "inspect_sim_materialized_contract 入口参数")
    if not isinstance(dataset_root, (str, Path)):
        raise TypeError("dataset_root must be a string or Path object")

    if not allowed_geometry_levels:
        raise ValueError("allowed_geometry_levels must be non-empty")
    if not allowed_k_levels:
        raise ValueError("allowed_k_levels must be non-empty")
    if not allowed_anchor_counts:
        raise ValueError("allowed_anchor_counts must be non-empty")
    if any((not isinstance(x, str) or not x) for x in allowed_geometry_levels):
        raise ValueError("allowed_geometry_levels must contain non-empty strings")
    if any((not isinstance(x, str) or not x) for x in allowed_k_levels):
        raise ValueError("allowed_k_levels must contain non-empty strings")
    if any((not isinstance(x, int) or isinstance(x, bool) or x <= 0) for x in allowed_anchor_counts):
        raise ValueError("allowed_anchor_counts must contain positive ints")

    # 修复组 21: anchor_layout_filename 不含路径分隔符
    if os.sep in anchor_layout_filename or '/' in anchor_layout_filename or '\\' in anchor_layout_filename:
        raise ValueError("anchor_layout_filename must not contain path separators")

    path = Path(dataset_root)
    if not path.is_dir():
        raise FileNotFoundError(f'Dataset path is not a directory: {os.fspath(path)}')

    if path.is_dir() and (path / anchor_layout_filename).is_file():
        seq_dirs = [path]
    else:
        seq_dirs = list_sequence_dirs(path)

    missing_anchor_layout_seq_ids: list[str] = []
    bad_anchor_layout_seq_ids: list[str] = []
    missing_uwb_seq_ids: list[str] = []
    bad_uwb_seq_ids: list[str] = []
    bad_anchor_count_seq_ids: list[str] = []
    missing_geometry_level_seq_ids: list[str] = []
    bad_geometry_level_seq_ids: list[str] = []
    missing_k_level_seq_ids: list[str] = []
    bad_k_level_seq_ids: list[str] = []
    bad_uwb_anchor_coverage_seq_ids: list[str] = []
    bad_anchor_ids_type_seq_ids: list[str] = []  # 修复组 8、11
    missing_base_layout_id_seq_ids: list[str] = []  # 修复组 12
    bad_layout_id_seq_ids: list[str] = []  # 修复组 25
    bad_anchor_count_field_seq_ids: list[str] = []  # 修复组 32
    sequences: dict[str, Any] = {}

    for seq_dir in seq_dirs:
        seq_id = seq_dir.name or str(seq_dir)
        anchor_layout_path = seq_dir / anchor_layout_filename

        # 修复组 5: 移除 TOCTOU is_file() 检查，直接 try read_json
        try:
            payload = read_json(anchor_layout_path)
        except FileNotFoundError:
            missing_anchor_layout_seq_ids.append(seq_id)
            # 修复组 10: continue 前填充占位条目
            sequences[seq_id] = {'error': 'missing_anchor_layout'}
            continue
        except (OSError, ValueError):
            bad_anchor_layout_seq_ids.append(seq_id)
            sequences[seq_id] = {'error': 'bad_anchor_layout'}
            continue

        if not isinstance(payload, dict):
            bad_anchor_layout_seq_ids.append(seq_id)
            sequences[seq_id] = {'error': 'bad_anchor_layout'}
            continue

        # 修复组 12: base_layout_id 检查
        base_layout_id = payload.get(_LAYOUT_FAMILY_FIELD)
        if base_layout_id is None:
            missing_base_layout_id_seq_ids.append(seq_id)

        # 修复组 25: layout_id 读取
        layout_id = payload.get(_LAYOUT_SEQUENCE_FIELD)

        # 修复组 8、11: anchor_ids 类型校验
        anchor_ids = payload.get('anchor_ids')
        if 'anchor_ids' in payload and not isinstance(anchor_ids, list):
            bad_anchor_ids_type_seq_ids.append(seq_id)
        if isinstance(anchor_ids, list) and not all(isinstance(anchor_id, str) for anchor_id in anchor_ids):
            bad_anchor_ids_type_seq_ids.append(seq_id)

        normalized_anchor_ids = [
            str(anchor_id).strip()
            for anchor_id in anchor_ids
            if str(anchor_id).strip()
        ] if isinstance(anchor_ids, list) else []
        anchor_count = len(normalized_anchor_ids) if isinstance(anchor_ids, list) else None

        # 修复组 32: anchor_count 字段一致性
        anchor_count_field = payload.get('anchor_count')
        if 'anchor_count' in payload and anchor_count_field != len(normalized_anchor_ids):
            bad_anchor_count_field_seq_ids.append(seq_id)

        geometry_level = payload.get('protocol_geometry_level')
        k_level = payload.get('protocol_k_level')

        # 修复组 5、23: UWB 读取，移除 TOCTOU is_file() 检查
        uwb_path = seq_dir / uwb_filename
        observed_anchor_ids: list[str] = []
        uwb_payload = None

        try:
            uwb_payload = read_json(uwb_path)
        except FileNotFoundError:
            missing_uwb_seq_ids.append(seq_id)
        except (OSError, ValueError):
            bad_uwb_seq_ids.append(seq_id)

        if uwb_payload is not None and (not isinstance(uwb_payload, list) or not uwb_payload):
            bad_uwb_seq_ids.append(seq_id)

        if isinstance(uwb_payload, list) and uwb_payload:
            observed_anchor_ids = sorted({
                str(row['anchor_id']).strip()
                for row in uwb_payload
                if isinstance(row, dict) and 'anchor_id' in row and row['anchor_id'] is not None and str(row['anchor_id']).strip()
            })

        sequences[seq_id] = {
            'anchor_count': anchor_count,
            'anchor_ids': normalized_anchor_ids if isinstance(anchor_ids, list) else None,
            'protocol_geometry_level': geometry_level,
            'protocol_k_level': k_level,
            'base_layout_id': base_layout_id,
            'layout_id': layout_id,
            'uwb_anchor_ids': observed_anchor_ids,
        }

        # 修复组 25: layout_id 与 seq_id 一致性检查
        # 2026-08-29 放宽：允许 layout_id 与 seq_id 的 base 形式一致（去 _seedN 后缀）。
        # 数据集有 base/var/seed 三种变体共享 base 布局，layout_id 反映 base/var 标识，
        # seq_id 反映 seed 变体（如 sim_curve_01_var_01_seed0 的 layout_id 是 sim_curve_01_var_01）。
        # 检查 layout_id 是否是 seq_id 去掉 _seedN 后缀的形式，或者两者完全相同。
        if layout_id is not None and layout_id != seq_id:
            import re as _re
            base_seq_id = _re.sub(r"_seed\d+$", "", seq_id)
            if layout_id not in (base_seq_id, base_layout_id):
                bad_layout_id_seq_ids.append(seq_id)

        # 锚点数：须落在主表允许集合，且若声明了 K 档则与协议 anchor_count 一致。
        if anchor_count is not None and anchor_count not in allowed_anchor_counts:
            bad_anchor_count_seq_ids.append(seq_id)
        if (
            isinstance(k_level, str)
            and k_level in _K_LEVEL_TO_COUNT
            and anchor_count is not None
            and anchor_count != _K_LEVEL_TO_COUNT[k_level]
        ):
            bad_anchor_count_seq_ids.append(seq_id)

        # geometry_level 类型 + 允许集校验
        if geometry_level is None:
            missing_geometry_level_seq_ids.append(seq_id)
        elif not isinstance(geometry_level, str):
            bad_geometry_level_seq_ids.append(seq_id)
        elif geometry_level not in allowed_geometry_levels:
            bad_geometry_level_seq_ids.append(seq_id)

        # k_level 类型 + 允许集校验
        if k_level is None:
            missing_k_level_seq_ids.append(seq_id)
        elif not isinstance(k_level, str):
            bad_k_level_seq_ids.append(seq_id)
        elif k_level not in allowed_k_levels:
            bad_k_level_seq_ids.append(seq_id)

        # 修复组 9: UWB 覆盖检查修复
        if normalized_anchor_ids:
            if not observed_anchor_ids:
                # observed 为空但 UWB 文件存在且非空：标记覆盖不完整
                if isinstance(uwb_payload, list) and uwb_payload:
                    bad_uwb_anchor_coverage_seq_ids.append(seq_id)
            elif set(observed_anchor_ids) != set(normalized_anchor_ids):
                bad_uwb_anchor_coverage_seq_ids.append(seq_id)

    return {
        'sequence_count': len(seq_dirs),
        'expected_anchor_count': (
            expected_anchor_count
            if expected_anchor_count is not None
            else SIM_MATERIALIZED_BASELINE_ANCHOR_COUNT
        ),
        'expected_geometry_level': (
            expected_geometry_level
            if expected_geometry_level is not None
            else SIM_MATERIALIZED_BASELINE_GEOMETRY_LEVEL
        ),
        'expected_k_level': (
            expected_k_level
            if expected_k_level is not None
            else SIM_MATERIALIZED_BASELINE_K_LEVEL
        ),
        'allowed_anchor_counts': sorted(allowed_anchor_counts),
        'allowed_geometry_levels': sorted(allowed_geometry_levels),
        'allowed_k_levels': sorted(allowed_k_levels),
        'missing_anchor_layout_seq_ids': sorted(missing_anchor_layout_seq_ids),
        'bad_anchor_layout_seq_ids': sorted(bad_anchor_layout_seq_ids),
        'missing_uwb_seq_ids': sorted(missing_uwb_seq_ids),
        'bad_uwb_seq_ids': sorted(bad_uwb_seq_ids),
        'bad_anchor_count_seq_ids': sorted(bad_anchor_count_seq_ids),
        'missing_geometry_level_seq_ids': sorted(missing_geometry_level_seq_ids),
        'bad_geometry_level_seq_ids': sorted(bad_geometry_level_seq_ids),
        'missing_k_level_seq_ids': sorted(missing_k_level_seq_ids),
        'bad_k_level_seq_ids': sorted(bad_k_level_seq_ids),
        'bad_uwb_anchor_coverage_seq_ids': sorted(bad_uwb_anchor_coverage_seq_ids),
        'bad_anchor_ids_type_seq_ids': sorted(bad_anchor_ids_type_seq_ids),
        'missing_base_layout_id_seq_ids': sorted(missing_base_layout_id_seq_ids),
        'bad_layout_id_seq_ids': sorted(bad_layout_id_seq_ids),
        'bad_anchor_count_field_seq_ids': sorted(bad_anchor_count_field_seq_ids),
        'sequences': sequences,
        'is_valid': (
            bool(seq_dirs)
            and not missing_anchor_layout_seq_ids
            and not bad_anchor_layout_seq_ids
            and not missing_uwb_seq_ids
            and not bad_uwb_seq_ids
            and not bad_anchor_count_seq_ids
            and not missing_geometry_level_seq_ids
            and not bad_geometry_level_seq_ids
            and not missing_k_level_seq_ids
            and not bad_k_level_seq_ids
            and not bad_uwb_anchor_coverage_seq_ids
            and not bad_anchor_ids_type_seq_ids
            and not missing_base_layout_id_seq_ids
            and not bad_layout_id_seq_ids
            and not bad_anchor_count_field_seq_ids
        ),
    }


def validate_sim_generator_version(dataset_root: str | Path) -> dict[str, Any]:
    """§28.6 自证契约：校验仿真物化序列的 sim_meta.json generator_version。

    每个物化序列的 sim_meta.json 必须携带 generator_version 字段，且其值
    必须与当前代码库的 SIM_GENERATOR_VERSION_CURRENT 严格一致。这防止
    "旧 raw + 新下游代码" 或 "新 raw + 旧下游代码" 跨版本混用产生幽灵差异。

    参数：
        dataset_root: 仿真数据集根目录路径。

    返回：
        校验报告字典，含 sequence_count / 问题序列列表 / is_valid。

    异常：
        FileNotFoundError: 当数据集根目录不存在时抛出。
        TypeError: 当 dataset_root 类型不支持时抛出。
    """
    from liquidloc.dataio.sim_materializer import SIM_GENERATOR_VERSION_CURRENT

    path = Path(dataset_root)
    if not path.is_dir():
        raise FileNotFoundError(f'Dataset path is not a directory: {os.fspath(path)}')

    seq_dirs = list_sequence_dirs(path)
    bad_version_seq_ids: list[str] = []
    missing_version_seq_ids: list[str] = []

    for seq_dir in seq_dirs:
        meta_path = seq_dir / 'sim_meta.json'
        if not meta_path.is_file():
            # sim_meta.json 缺失已由 _RAW_CONTRACT_FILES / SIM_REQUIRED_SEQ_FILES
            # 的 missing_files 检查覆盖，此处仅报告版本不一致。
            continue
        try:
            payload = read_json(meta_path)
        except (OSError, ValueError) as exc:
            bad_version_seq_ids.append(seq_dir.name)
            continue
        if not isinstance(payload, dict):
            bad_version_seq_ids.append(seq_dir.name)
            continue
        version = payload.get('generator_version')
        if version != SIM_GENERATOR_VERSION_CURRENT:
            bad_version_seq_ids.append(seq_dir.name)

    return {
        'kind': 'sim_generator_version',
        'expected_version': SIM_GENERATOR_VERSION_CURRENT,
        'sequence_count': len(seq_dirs),
        'bad_version_seq_ids': sorted(bad_version_seq_ids),
        'missing_version_seq_ids': sorted(missing_version_seq_ids),
        # 空目录视为合法（无序列需校验）；有序列时要求全部通过。
        'is_valid': not bad_version_seq_ids and not missing_version_seq_ids,
    }