"""实验场景采样与任务展开模块。

职责：
    把实验配置（experiment_cfg）展开成可执行的场景任务列表（scene tasks），
    每个任务携带完整的五轴参数（A/N/V/K/M）、场景 ID 和协议展开参数。

    采样过程严格遵循冻结场景轴协议（scene_axis_protocol），保证不同运行模式下
    产出的场景任务集合是确定性的、可复现的。

上游依赖：
    - liquidloc.protocol.experiment_gates      — 运行模式归一化（quick/full 等）
    - liquidloc.protocol.scene_axis_protocol   — 冻结场景轴协议加载与参数挂载

下游调用者：
    - liquidloc.pipelines.core_pipeline        — 核心流水线在场景构建阶段调用 sample_scenes
    - scripts / smoke tests                    — 脚本在入口处展开实验配置

核心变量：
    - _AXES                                    — 五轴标准顺序 ('A', 'N', 'V', 'K', 'M')

关键设计决策：
    - primary_axis 决定采样策略：单轴扫描、双轴交叉、退化组合、公开序列等，
      每种策略有独立的展开逻辑和参数校验。
    - frozen_axes 中未指定的轴会从协议配置中取默认等级（第一个定义的等级）。
    - 重复展开（repeats）和截断（max_sequences）由 mode_overrides 控制，
      保证 quick 模式下任务数可控。
    - 公开序列模式（public_sequence_category）不支持重复展开，
      因为公开验证要求每次运行覆盖完全相同的序列集合。
"""

from __future__ import annotations  # 允许类型注解使用现代写法。

from collections.abc import Sequence  # 用于判断输入是否为序列类型。
from copy import deepcopy  # 深拷贝，保证展开任务之间不共享可变引用。
from itertools import product  # 笛卡尔积，用于双轴交叉展开。
from typing import Any  # 用于宽松类型标注。

from liquidloc.dataio.registry.public_dataset_registry import resolve_public_eval_seq_ids  # 统一解析公开评测序列合同。
from liquidloc.common.validation import is_bool_like, is_integer, is_string_like  # 统一布尔/整数/字符串类型校验函数。
from liquidloc.common.validation import is_real as _is_real  # 统一实数类型校验函数。
from liquidloc.protocol.experiment_gates import (  # 运行模式归一化与公开基准闸门。
    get_public_benchmark_allowed_datasets,
    get_public_benchmark_frozen_eval_split,
    normalize_public_dataset_name,
    normalize_run_mode,
)
from liquidloc.protocol.scene_axis_protocol import SCENE_AXES, AXIS_METADATA_KEYS, attach_scene_parameters, get_nominal_levels, load_scene_axis_protocol  # 场景轴协议加载与参数挂载。

_AXES = SCENE_AXES  # 五轴标准顺序：异步/非视距/视觉/锚点数/缺失模态，从协议层导入避免漂移。


def _coerce_python_native(value: Any) -> Any:
    """将 numpy 标量转为 Python 原生类型，防止 numpy 类型泄漏到下游。

    Args:
        value: 任意值，numpy 整数/浮点/布尔标量会被转为 Python 原生类型。

    Returns:
        Python 原生类型的值，非 numpy 标量原样返回。
    """
    if is_bool_like(value):  # numpy.bool_ 必须在 is_integer 之前检查，因为 np.bool_ 是 np.integer 的子类。
        return bool(value)
    if is_integer(value):
        return int(value)
    if _is_real(value) and not isinstance(value, int):
        return float(value)
    return value


def _as_list(value: Any) -> list[Any]:
    """将标量或非字符串序列统一归一化为列表。

    方便后续统一用列表语义处理配置值，无论配置写的是单个值还是列表。

    Args:
        value: 输入值，可以是 None、列表、其他序列或标量。

    Returns:
        list[Any]: 归一化后的列表。None 返回空列表，列表原样返回，
            其他非字符串序列转为列表，标量包装成单元素列表。
    """
    if value is None:  # None 表示未配置，返回空列表。
        return []
    if isinstance(value, list):  # 已经是列表就直接用，避免不必要的拷贝。
        return [_coerce_python_native(v) for v in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):  # 元组等序列类型转成列表，但排除字符串。
        return [_coerce_python_native(v) for v in value]
    return [_coerce_python_native(value)]  # 标量值包装成单元素列表，统一后续处理逻辑。


def _normalize_sweep_cfgs(sweep_cfgs: dict[str, Any]) -> dict[str, Any]:
    """把 sweep 配置规整成 sample_scenes 可直接消费的覆盖格式。

    当前仓内 `configs/sweeps/*.yaml` 使用：
        levels:
          - {name: A0, strength: 0.0}
          - {name: A1, strength: 0.33}

    而采样器运行时真正需要的是：
        levels: [{name: A0, strength: 0.0}, {name: A1, strength: 0.33}, ...]

    这里仅做最小兼容归一化：当 levels 是映射列表且项里带 name 字段时，
    保留原始 dict（含 name 和 strength 等键）供上游可选消费，
    同时确保 name 字段为非空字符串。
    """
    normalized_cfg = dict(sweep_cfgs)
    raw_levels = normalized_cfg.get('levels')
    if not isinstance(raw_levels, Sequence) or isinstance(raw_levels, (str, bytes)):
        return normalized_cfg
    normalized_levels: list[Any] = []
    for level_entry in raw_levels:
        if isinstance(level_entry, dict):
            level_name = level_entry.get('name')
            if not is_string_like(level_name) or not str(level_name).strip():
                raise TypeError("sweep_cfgs.levels[] mapping entries must define a non-empty string 'name'")
            # 保留原始 dict，确保 name 为归一化后的值
            normalized_entry = dict(level_entry)
            normalized_entry['name'] = str(level_name).strip()
            normalized_levels.append(normalized_entry)
        else:
            normalized_levels.append(level_entry)
    normalized_cfg['levels'] = normalized_levels
    return normalized_cfg


def _level_name(level_entry: Any) -> str:
    """从等级条目中提取等级名称字符串。

    等级条目可以是字符串（如 'A0'）或包含 'name' 键的字典
    （如 {'name': 'A0', 'strength': 0.33}）。
    当 _normalize_sweep_cfgs 保留完整 dict 后，下游消费者
    需要通过此函数提取 name 字段作为轴等级值。

    Args:
        level_entry: 等级条目，字符串或含 'name' 键的字典。

    Returns:
        等级名称字符串（去除首尾空白）。

    Raises:
        TypeError: 当字典条目缺少非空字符串 'name' 键，
            或条目为 None/非字符串类型时抛出。
    """
    if level_entry is None:  # None 会被 str() 静默转为 'None'，必须在入口拒绝。
        raise TypeError("level entry must not be None")
    if isinstance(level_entry, dict):
        name = level_entry.get('name')
        if is_string_like(name) and str(name).strip():
            return str(name).strip()
        raise TypeError("dict level entry must contain a non-empty string 'name'")
    if not is_string_like(level_entry):  # 非字符串类型（如 int）会被 str() 静默转换，拒绝以暴露配置错误。
        raise TypeError(f"level entry must be a string or dict, got {type(level_entry).__name__}")
    return str(level_entry).strip()  # 显式转 Python str 并去空白，兼容 numpy.str_。


def _build_scene_id(axes: dict[str, Any]) -> str:
    """根据五轴参数构建稳定的场景标识符。

    五轴参数齐全时生成 S(A,N,V,K,M) 格式的紧凑标识。
    不齐全时抛出 ValueError，避免生成与下游 decode_scene 不兼容的
    key=value 格式导致管线静默失败。

    Args:
        axes: 轴参数字典，键为轴名（如 'A'、'N'），值为等级名。

    Returns:
        str: 场景标识字符串，格式为 S(A,N,V,K,M)。
        注意：2026-08-31 G 轴并入 K 轴，scene_code 升级为 A/N/V/K/M 五轴。

    Raises:
        ValueError: 五轴参数不齐全、某轴值为 None，或 M 轴非正常等级时抛出。
    """
    missing_axes = [axis for axis in _AXES if axis not in axes]  # 找出缺失的轴。
    if missing_axes:  # 五轴不齐全时不能生成合法的场景标识。
        raise ValueError(f'_build_scene_id requires all five axes, missing: {missing_axes}')
    none_axes = [axis for axis in _AXES if axes[axis] is None]  # 找出值为 None 的轴。
    if none_axes:  # None 值会被 str() 静默转为 'None'，必须在入口拒绝。
        raise ValueError(f'_build_scene_id axes must not be None: {none_axes}')
    # M 轴防御性校验：sampler 暂不支持多值 M。若 M 不是正常等级，
    # 不同 M 值将映射到相同 scene_id，引发冲突。此处显式拒绝。
    nominal_m = get_nominal_levels().get('M')
    if nominal_m is not None and str(axes['M']) != str(nominal_m):
        raise ValueError(
            f'_build_scene_id M axis must be nominal level ({nominal_m}) due to '
            f'sampler not supporting multi-value M expansion; got {axes["M"]!r}. '
            f'Multi-value M expansion requires extending scene_sampler logic.'
        )
    from liquidloc.protocol.scene_schema import SceneSpec, encode_scene
    spec = SceneSpec(A_level=str(axes['A']), N_level=str(axes['N']), V_level=str(axes['V']), K_value=str(axes['K']), M_level=str(axes['M']))
    return encode_scene(spec)


def _default_axis_level(axis: str, protocol_cfg: dict[str, Any]) -> str:
    """从冻结场景协议中解析指定轴的默认等级。

    默认等级取 get_nominal_levels() 中该轴的正常等级名，
    与桥接合同和训练管线中的"正常场景"判断语义一致。
    A/N/V/G 轴取第一个等级，K 轴取中间值。

    Args:
        axis: 轴名（如 'A'、'N'）。
        protocol_cfg: 冻结场景轴协议配置字典。

    Returns:
        str: 默认等级名。

    Raises:
        ValueError: 协议配置中缺少该轴的定义时抛出。
    """
    nominal = get_nominal_levels()  # 从协议动态获取正常等级名。
    if axis in nominal:
        return nominal[axis]
    # 回退：如果轴不在 nominal 中（理论上不会发生），从协议配置中手动查找。
    axis_cfg = dict((protocol_cfg.get('axes') or {}).get(axis) or {})
    if not axis_cfg:
        raise ValueError(f'scene axis protocol missing definitions for axis {axis}')
    for key in axis_cfg:
        if key not in AXIS_METADATA_KEYS:
            return key
    raise ValueError(f'scene axis protocol has no valid level for axis {axis}')


def _coerce_positive_int(value: Any, *, name: str) -> int:
    """校验并转换正整数配置值。

    拒绝 bool（虽然 bool 是 Integral 的子类），拒绝零和负数，
    保证 repeats 和 max_sequences 等参数语义正确。

    Args:
        value: 待校验的值。
        name: 参数名，用于错误消息。

    Returns:
        int: 校验后的正整数。

    Raises:
        TypeError: 值不是整数或是 bool 时抛出。
        ValueError: 值不是正数时抛出。
    """
    if not is_integer(value):  # bool 虽然是 Integral 子类，但不是合法的正整数语义。
        raise TypeError(f'{name} must be a positive integer')
    int_value = int(value)  # 统一转成 Python int，兼容 numpy 整数等。
    if int_value <= 0:  # 正整数必须大于零。
        raise ValueError(f'{name} must be positive when provided')
    return int_value


def _normalize_non_blank_string(value: Any, *, name: str) -> str:
    """校验并归一化非空字符串配置值。

    去除首尾空白后检查是否为空，用于 dataset_name、split 等必须
    有实际内容的配置字段。

    Args:
        value: 待校验的值。
        name: 参数名，用于错误消息。

    Returns:
        str: 去除首尾空白后的非空字符串。

    Raises:
        TypeError: 值不是字符串时抛出。
        ValueError: 值为空白字符串时抛出。
    """
    if not is_string_like(value):  # 必须是字符串类型。
        raise TypeError(f'{name} must be a string')
    normalized = str(value).strip()  # 去除首尾空白。
    if not normalized:  # 空白字符串没有实际内容。
        raise ValueError(f'{name} must not be blank')
    if '\x00' in normalized:  # 拒绝含空字节的字符串，防止注入和路径穿越风险（与 public_dataset_registry._normalize_public_seq_id_list 对齐）。
        raise ValueError(f'{name} must not contain null bytes')
    return normalized


def _normalize_primary_axis(primary_axis: Any) -> str | list[str]:
    """归一化 primary_axis 配置值为字符串或字符串列表。

    支持单轴字符串（如 'A'）和多轴列表（如 ['G', 'K']），
    统一后续分支判断逻辑。numpy.str_ 会被显式转为 Python str，
    防止 numpy 类型泄漏到下游。

    Args:
        primary_axis: 原始配置值，可以是字符串或字符串序列。

    Returns:
        str | list[str]: 归一化后的主轴标识。
    """
    if is_string_like(primary_axis):  # 单轴直接返回，显式转 Python str 防止 numpy.str_ 泄漏。
        return str(primary_axis)
    # bytearray 和 memoryview 是 Sequence 子类但不应被逐字节迭代。
    if isinstance(primary_axis, Sequence) and not isinstance(primary_axis, (str, bytes, bytearray, memoryview)):
        return [str(item) for item in primary_axis]
    return primary_axis  # 其他类型原样返回，后续校验会捕获非法值。


def _single_task(
    axes: dict[str, Any],
    *,
    dataset_name: str | None = None,
    seq_id: str | None = None,
    index: int = 0,
    protocol_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """实例化单个场景任务字典。

    根据轴参数生成一个完整的场景任务，包含任务 ID、轴参数、场景 ID
    和可选的协议展开参数、数据集名和序列 ID。

    Args:
        axes: 五轴参数字典，键为轴名，值为等级名。
        dataset_name: 可选的数据集名称，公开序列模式使用。
        seq_id: 可选的序列 ID，公开序列模式使用。
        index: 任务编号，用于生成 task_id。
        protocol_cfg: 冻结场景轴协议配置，用于挂载 scene_parameters。

    Returns:
        dict[str, Any]: 场景任务字典，包含 task_id、axes、scene_id
            以及可选的 scene_parameters、dataset_name、seq_id。
    """
    task: dict[str, Any] = {
        'task_id': f'scene_{index:04d}',  # 用四位数编号保证排序稳定，与 prepared_inputs.py 对齐。
        'axes': dict(axes),  # 拷贝轴参数，避免共享引用。
        'scene_id': _build_scene_id(axes),  # 生成稳定的场景标识。
    }
    if all(axis in axes for axis in _AXES):  # 五轴齐全时才能挂载协议参数。
        task['scene_parameters'] = attach_scene_parameters(axes, protocol_cfg).to_dict()  # 把协议展开参数写入任务（纯 dict）。
    if dataset_name is not None:  # 公开序列模式需要记录数据集名。
        task['dataset_name'] = dataset_name
    if seq_id is not None:  # 公开序列模式需要记录序列 ID。
        task['seq_id'] = seq_id
    return task


def _with_repeat_and_limit(
    base_tasks: list[dict[str, Any]],
    *,
    repeats: int | None,
    max_sequences: int | None,
) -> list[dict[str, Any]]:
    """对基础任务列表施加序列上限和重复展开。

    `max_sequences` 的科学语义是“最多保留多少个基础序列/场景任务”，
    而不是“重复展开后的总任务数上限”。因此应先裁剪基础任务列表，
    再对保留下来的任务做 repeats 次重复展开，避免把同一批场景的重复组切残。

    Args:
        base_tasks: 基础场景任务列表。
        repeats: 重复次数，None 或未提供时默认为 1。
        max_sequences: 最大基础序列数阈值，None 表示不截断。

    Returns:
        list[dict[str, Any]]: 经过序列裁剪和重复展开后的任务列表。
    """
    repeat_count = _coerce_positive_int(repeats, name='mode_overrides.repeats') if repeats is not None else 1  # 未指定时默认不重复。
    limited_base_tasks = base_tasks  # 默认保留全部基础任务。
    if max_sequences is not None:  # 先限制基础序列数，再做重复展开。
        max_count = _coerce_positive_int(max_sequences, name='mode_overrides.max_sequences')  # 校验基础序列上限。
        limited_base_tasks = base_tasks[:max_count]  # 按基础任务维度截断，避免切残重复组。
    expanded_tasks: list[dict[str, Any]] = []  # 收集展开后的任务。
    task_index = 0  # 全局任务编号，保证 task_id 唯一。
    for repeat_index in range(repeat_count):  # 每次重复生成一组任务。
        repeat_id = f'repeat_{repeat_index:04d}'  # 重复编号，用于区分同一场景的多次运行。
        for base_task in limited_base_tasks:  # 对每个保留下来的基础任务做一次深拷贝。
            task = deepcopy(base_task)  # 深拷贝避免共享可变引用。
            task['task_id'] = f'scene_{task_index:04d}'  # 重新编号保证全局唯一，4 位数对齐 prepared_inputs.py。
            task['repeat_id'] = repeat_id  # 标记属于哪次重复。
            task['scene_variant_id'] = f"{task['scene_id']}::{repeat_id}"  # 场景变体标识，区分同一场景的不同重复。
            expanded_tasks.append(task)
            task_index += 1
    return expanded_tasks  # 重复展开后的总任务数由基础序列数和 repeats 共同决定。


def _apply_mode_limits(
    base_tasks: list[dict[str, Any]],
    mode_cfg: dict[str, Any],
    *,
    allow_repeats: bool = True,
) -> list[dict[str, Any]]:
    """根据运行模式配置施加重复和截断控制。

    当 allow_repeats=False 时（如公开序列模式），只做截断不做重复展开，
    因为公开验证要求每次运行覆盖完全相同的序列集合。

    Args:
        base_tasks: 基础场景任务列表。
        mode_cfg: 运行模式配置字典，可包含 repeats 和 max_sequences。
        allow_repeats: 是否允许重复展开，默认为 True。

    Returns:
        list[dict[str, Any]]: 经过模式限制后的任务列表。
    """
    if not allow_repeats:  # 公开序列模式不支持重复。
        max_sequences = mode_cfg.get('max_sequences')  # 只取截断阈值。
        if max_sequences is None:  # 未指定截断时直接返回。
            return base_tasks
        max_count = _coerce_positive_int(max_sequences, name='mode_overrides.max_sequences')
        return base_tasks[:max_count]  # 只截断不重复。
    return _with_repeat_and_limit(  # 非公开模式走完整的重复+截断逻辑。
        base_tasks,
        repeats=mode_cfg.get('repeats'),  # 从模式配置中取重复次数。
        max_sequences=mode_cfg.get('max_sequences'),  # 从模式配置中取截断阈值。
    )


def _resolve_base_axes(
    frozen_axes: dict[str, Any],
    protocol_cfg: dict[str, Any],
    *,
    allow_multi_axes: set[str] | None = None,
) -> dict[str, Any]:
    """解析五轴基础参数，校验单轴约束并填充默认等级。

    对于 frozen_axes 中未指定的轴，从协议配置中取默认等级；
    对于允许多值展开的轴（如 K 轴在 runtime_profile 模式下），
    只取第一个值作为基础参数，多值展开由上层逻辑处理。

    Args:
        frozen_axes: 调用方指定的冻结轴参数字典。
        protocol_cfg: 冻结场景轴协议配置字典。
        allow_multi_axes: 允许多值展开的轴名集合，默认为空。

    Returns:
        dict[str, Any]: 基础轴参数字典，每个轴只有一个等级值。

    Raises:
        ValueError: 不允许多值的轴传入了多个等级时抛出。
    """
    allowed_multi_axes = set(allow_multi_axes or ())  # 默认没有轴允许多值。
    base_axes: dict[str, Any] = {}  # 收集基础轴参数。
    for axis in _AXES:  # 按标准顺序遍历五轴。
        axis_levels = _as_list(frozen_axes.get(axis))  # 把配置值统一转成列表。
        if len(axis_levels) > 1 and axis not in allowed_multi_axes:  # 不允许多值时只能指定一个等级。
            raise ValueError(f'frozen_axes.{axis} must be a single level in this sampling mode')
        # 使用 _level_name 提取等级名，与 runtime_profile 和单轴扫描模式保持一致，
        # 确保 dict 形式的等级条目（如 {'name': 'A0', 'strength': 0.33}）被正确解析。
        base_axes[axis] = _level_name(axis_levels[0]) if axis_levels else _default_axis_level(axis, protocol_cfg)  # 有值取第一个并提取名称，没值取默认。
    return base_axes


def sample_scenes(experiment_cfg: dict[str, Any], sweep_cfgs: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """将实验配置展开为场景任务列表。

    根据实验配置中的 primary_axis 决定采样策略，支持以下模式：

    1. **单轴扫描**（primary_axis 为 A/N/V/G/K 之一）：
       在指定轴上遍历所有等级，其余轴冻结。
    2. **退化组合**（target_degradation_bundle / dual_degradation_bundle）：
       五轴全展开的笛卡尔积，用于系统性退化实验。
    3. **公开序列**（public_sequence_category）：
       按数据集和序列 ID 列表展开，不支持重复。
    4. **安全模式/消融**（safe_mode_behavior 等）：
       单任务模式，只生成一个场景任务。
    5. **运行时配置**（runtime_profile）：
       在 K 轴上遍历不同等级，其余轴冻结。
    6. **K 轴扫描**（'K'）：
       锚点数量的单轴扫描。

    Args:
        experiment_cfg: 实验配置字典，必须包含 primary_axis，
            可包含 frozen_axes、mode、mode_overrides 等字段。
        sweep_cfgs: 可选的扫描配置覆盖字典，会合并到 mode_overrides 中。

    Returns:
        list[dict[str, Any]]: 展开后的场景任务列表，每个任务包含
            task_id、axes、scene_id 以及可选的 scene_parameters 等。

    Raises:
        ValueError: 配置不合法（如 primary_axis 缺失、frozen_axes 键不支持、
            等级未配置等）时抛出。
        TypeError: sweep_cfgs 不是映射类型时抛出。
    """
    # 第 9 轮审查 MEDIUM-1 修复：sample_scenes 是场景采样入口函数，每实验调用一次。
    # 大规模实验编排（e1-e12 全量跑）下入口 print_dict 产生大量日志，违反工程规范
    # "Recursive functions and entry points should avoid print_dict calls"。
    if not isinstance(experiment_cfg, dict) or not experiment_cfg:  # 实验配置必须是非空字典。
        raise ValueError('experiment_cfg must be a non-empty mapping')

    primary_axis = _normalize_primary_axis(experiment_cfg.get('primary_axis'))  # 归一化主轴配置。
    frozen_axes = dict(experiment_cfg.get('frozen_axes') or {})  # 冻结轴参数，默认为空字典。
    mode = normalize_run_mode(experiment_cfg.get('mode'), default_mode='full')  # 归一化运行模式，默认为 full。
    mode_cfg = dict((experiment_cfg.get('mode_overrides') or {}).get(mode) or {})  # 取当前模式对应的覆盖配置。
    if sweep_cfgs is not None:  # 扫描配置会覆盖模式配置中的同名键。
        if not isinstance(sweep_cfgs, dict):
            raise TypeError('sweep_cfgs must be a mapping when provided')
        mode_cfg.update(_normalize_sweep_cfgs(dict(sweep_cfgs)))  # 把扫描配置归一化后合并进来。
    protocol_cfg = load_scene_axis_protocol(experiment_cfg.get('scene_axis_protocol_path'))  # 加载冻结场景轴协议。
    if not primary_axis:  # primary_axis 是必需字段。
        raise ValueError('experiment_cfg.primary_axis is required')

    # 校验 frozen_axes 中是否包含不支持的键。
    allowed_meta_axes = {'dataset', 'split'}  # 元信息轴，仅公开序列模式使用。
    unknown_frozen_axes = sorted(
        axis for axis in frozen_axes if axis not in _AXES and axis not in allowed_meta_axes
    )  # 找出既不是五轴也不是元信息的键。
    if unknown_frozen_axes:  # 不支持的键说明配置写错了。
        raise ValueError(
            f'Unsupported frozen_axes keys: {unknown_frozen_axes}; '
            f'frozen_axes 仅支持 A/N/V/K/M 五个五轴档位协议键（plus dataset/split 元键）'
        )

    # 校验公开序列专用键不能出现在非公开模式中。
    public_only_frozen_axes = sorted(axis for axis in frozen_axes if axis in {'dataset', 'split'})  # 公开模式专用键。
    if primary_axis != 'public_sequence_category' and public_only_frozen_axes:  # 非公开模式不能使用这些键。
        raise ValueError(
            'scene-axis sampling supports only frozen_axes keys A/N/V/K/M outside public_sequence_category; '
            f'got public-only keys: {public_only_frozen_axes}'
        )

    # ---- 退化组合模式：五轴全展开的笛卡尔积 ----
    if primary_axis in ('target_degradation_bundle', 'dual_degradation_bundle'):
        # 退化组合要求所有五轴都显式指定，避免隐含默认值导致实验不可控。
        missing_axes = [axis for axis in _AXES if not _as_list(frozen_axes.get(axis))]
        if missing_axes:  # 缺少任何一轴都不能继续。
            raise ValueError(f'{primary_axis} requires explicit frozen_axes for ' + ', '.join(missing_axes))
        scene_tasks = [
            _single_task(
                {'A': a, 'N': n, 'V': v, 'K': k, 'M': m},  # 五轴参数齐全（A/N/V/K/M；G 从协议移除）。
                index=index,
                protocol_cfg=protocol_cfg,
            )
            for index, (a, n, v, k, m) in enumerate(
                product(  # 五轴笛卡尔积，遍历所有组合。
                    # 对每个 level 条目调用 _level_name() 提取名称，
                    # 避免 dict 形条目（如 {'name': 'A0', 'strength': 0.33}）穿透导致 _build_scene_id 崩溃。
                    [_level_name(lv) for lv in _as_list(frozen_axes.get('A'))],
                    [_level_name(lv) for lv in _as_list(frozen_axes.get('N'))],
                    [_level_name(lv) for lv in _as_list(frozen_axes.get('V'))],
                    [_level_name(lv) for lv in _as_list(frozen_axes.get('K'))],
                    [_level_name(lv) for lv in _as_list(frozen_axes.get('M'))],
                )
            )
        ]
        return _apply_mode_limits(scene_tasks, mode_cfg)  # 施加重复和截断。

    # ---- 公开序列模式：按数据集和序列 ID 展开 ----
    if primary_axis == 'public_sequence_category':
        # 公开序列模式只支持 dataset/split 两个冻结键，不支持场景轴键。
        scene_axis_frozen_keys = sorted(axis for axis in frozen_axes if axis in _AXES)
        if scene_axis_frozen_keys:  # 场景轴键在公开模式下不合法。
            raise ValueError(
                'public_sequence_category supports only frozen_axes keys dataset/split; '
                f'got scene-axis keys: {scene_axis_frozen_keys}'
            )
        # 公开基准数据集名称必须经过协议层归一化（小写+去空白）和白名单校验，
        # 防止 scene_sampler 绕过 experiment_gates.normalize_public_benchmark_request 的闸门。
        allowed_datasets = get_public_benchmark_allowed_datasets()
        raw_dataset_name = frozen_axes.get('dataset')
        if raw_dataset_name is None:  # 未指定时使用协议允许的第一个数据集（通常是 miluv）。
            dataset_name = allowed_datasets[0]
        else:
            dataset_name = normalize_public_dataset_name(raw_dataset_name)  # 协议层归一化（含 .lower()）。
        if dataset_name not in allowed_datasets:  # 白名单校验。
            raise ValueError(
                f'frozen_axes.dataset not allowed in public benchmark: {dataset_name!r}; '
                f'allowed: {sorted(allowed_datasets)}'
            )
        # 分割名必须与协议冻结的公开评测分割一致，禁止硬编码。
        frozen_eval_split = get_public_benchmark_frozen_eval_split()
        raw_split = frozen_axes.get('split')
        if raw_split is None and frozen_eval_split is not None:  # 仅在协议配置了冻结分割时才作为默认值填充。
            raw_split = frozen_eval_split
        if raw_split is not None:  # 用户指定了 split 才归一化校验。
            split = _normalize_non_blank_string(raw_split, name='frozen_axes.split')
            if frozen_eval_split is not None and split != frozen_eval_split:
                raise ValueError(
                    f'frozen_axes.split must be {frozen_eval_split!r} for public benchmark, got {split!r}'
                )
        else:
            split = None  # 协议未配置冻结分割且用户未指定时，允许为 None。
        # 公开序列模式需要使用合并后的 mode_cfg 来解析 seq_ids，
        # 避免 resolve_public_eval_seq_ids 使用原始 experiment_cfg 的 max_sequences
        # 而 _apply_mode_limits 使用合并后 mode_cfg 的 max_sequences 导致双重截断。
        merged_experiment_cfg = dict(experiment_cfg)
        merged_mode_overrides = dict(experiment_cfg.get('mode_overrides') or {})
        merged_mode_overrides[mode] = mode_cfg
        merged_experiment_cfg['mode_overrides'] = merged_mode_overrides
        seq_ids = resolve_public_eval_seq_ids(dataset_name, merged_experiment_cfg, mode)  # 公开序列统一从配置或注册表解析。
        scene_tasks = [
            {
                'task_id': f'public_{idx:04d}',  # 公开任务用 public_ 前缀区分，4 位数对齐。
                'dataset_name': dataset_name,
                'seq_id': seq_id,
                'scene_id': f'{dataset_name}:{seq_id}',  # 场景 ID 由数据集和序列组成。
                'axes': {'dataset': dataset_name, 'split': split},  # 公开模式没有五轴参数。
                'scene_parameters': {'axes': {}, 'flat': {'dataset': dataset_name, 'split': split}, 'axis_metadata': {}},  # 协议参数简化为元信息，含 axis_metadata 与 SceneParameters.to_dict() 对齐。
            }
            for idx, seq_id in enumerate(seq_ids)
        ]
        return _apply_mode_limits(scene_tasks, mode_cfg, allow_repeats=False)  # 公开模式不支持重复。

    # ---- 安全模式/消融：单任务模式 ----
    if primary_axis in (
        'safe_mode_behavior',
        'ablation_variant',
    ):
        # 这些模式只生成一个场景任务，轴参数全部冻结。
        base_axes = _resolve_base_axes(frozen_axes, protocol_cfg)
        return _apply_mode_limits(
            [_single_task(base_axes, index=0, protocol_cfg=protocol_cfg)],
            mode_cfg,
        )

    if primary_axis in ('modality_recovery_profile', 'phase_switch_profile'):
        # 这些 primary_axis 在实验配置中声明（e10/e11），但当前场景管道尚未实现
        # 对应的事件级模态恢复/相位切换执行路径。优雅降级为空任务列表，避免
        # 阻断整个 pipeline；调用方应感知无任务产出并在上层汇总中标记实验未运行。
        # 第 9 轮审查 MEDIUM-1 修复：graceful_degrade 分支的 print_dict 同步移除。
        return _apply_mode_limits([], mode_cfg)

    # ---- 运行时配置模式：在 K 轴上遍历不同等级 ----
    if primary_axis == 'runtime_profile':
        k_levels = _as_list(mode_cfg.get('levels')) or _as_list(frozen_axes.get('K'))  # K 轴等级列表。
        if not k_levels:  # 必须指定 K 轴等级。
            raise ValueError('No K levels configured for primary axis runtime_profile')
        base_axes = _resolve_base_axes(frozen_axes, protocol_cfg, allow_multi_axes={'K'})  # K 轴允许多值。
        scene_tasks = []
        for index, k_level in enumerate(k_levels):  # 遍历 K 轴等级。
            axes = dict(base_axes)  # 拷贝基础参数。
            axes['K'] = _level_name(k_level)  # 替换 K 轴等级（从 dict 中提取 name）。
            scene_tasks.append(_single_task(axes, index=index, protocol_cfg=protocol_cfg))
        return _apply_mode_limits(scene_tasks, mode_cfg)

    if isinstance(primary_axis, list):
        # 单轴列表模式（如 ['K']）→ 在 K 轴上遍历锚点数量等级。
        if len(primary_axis) != 1 or primary_axis[0] != 'K':
            raise ValueError(f'Unsupported primary_axis sequence: {primary_axis}')
        anchor_counts = _as_list(mode_cfg.get('levels')) or _as_list(frozen_axes.get('K'))  # 锚点数量列表。
        if not anchor_counts:  # 必须指定锚点数量。
            raise ValueError('No anchor_counts configured for primary axis K')
        # K 是主扫描轴，frozen_axes 中可能含多值列表（作为 levels 回退源），
        # 传入 allow_multi_axes 避免误报，_resolve_base_axes 只取首值作为基线。
        base_axes = _resolve_base_axes(frozen_axes, protocol_cfg, allow_multi_axes={'K'})  # 其余轴取默认值。
        scene_tasks = []
        for index, anchor_count in enumerate(anchor_counts):  # K 轴遍历。
            axes = dict(base_axes)
            axes['K'] = _level_name(anchor_count)  # 替换锚点数量（从 dict 中提取 name）。
            scene_tasks.append(_single_task(axes, index=index, protocol_cfg=protocol_cfg))
        return _apply_mode_limits(scene_tasks, mode_cfg)

    # ---- 单轴扫描模式：在指定轴上遍历等级 ----
    if primary_axis not in _AXES:  # 不支持的主轴类型。
        raise ValueError(f'Unsupported primary_axis: {primary_axis}')

    levels = _as_list(mode_cfg.get('levels')) or _as_list(frozen_axes.get(primary_axis))  # 主轴等级列表。
    if not levels:  # 必须指定主轴等级。
        raise ValueError(f'No levels configured for primary axis {primary_axis}')
    # 主扫描轴在 frozen_axes 中可能含多值列表（作为 levels 回退源），
    # 传入 allow_multi_axes 避免误报，_resolve_base_axes 只取首值作为基线。
    base_axes = _resolve_base_axes(frozen_axes, protocol_cfg, allow_multi_axes={primary_axis})  # 其余轴取默认值。
    scene_tasks = []
    for index, level in enumerate(levels):  # 遍历主轴等级。
        axes = dict(base_axes)
        axes[primary_axis] = _level_name(level)  # 替换主轴等级（从 dict 中提取 name）。
        scene_tasks.append(_single_task(axes, index=index, protocol_cfg=protocol_cfg))
    return _apply_mode_limits(scene_tasks, mode_cfg)  # 施加重复和截断。
