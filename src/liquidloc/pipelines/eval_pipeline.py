"""评估流水线。

这个模块负责把上游产生的 prediction bundle 转成评估指标、统计结果、
候选案例集和绘图输入。它是"预测结果 -> 指标 -> 统计 -> 选例 -> 产物"
这条链路的集中编排点，主要供实验评估、论文统计和结果审计使用。

统计编排逻辑已提取到 liquidloc.analysis.statistics_runner，
案例选择编排逻辑已提取到 liquidloc.analysis.case_selector_runner。
"""

from __future__ import annotations  # 允许在类型标注里引用尚未定义的类名。

import csv  # 用于把长表指标写成 CSV 文件。
import math  # 用于 isfinite 检查，避免 NaN/inf 数值漂移。
from collections.abc import Mapping, Sequence  # 用于判断映射型和序列型输入。
from copy import deepcopy  # 用于复制嵌套结构，避免共享引用。
from pathlib import Path  # 用于把路径字符串统一转成 Path。
from typing import Any  # 用于给动态结构写宽松类型注解。

from liquidloc.analysis.case_selector import select_cases  # 按选例规则从 case_catalog 里挑案例。
from liquidloc.analysis.case_selector_runner import build_empty_selected_cases, select_best_method, select_case_rules  # 案例选择编排：最优方法、分组规则、空容器。
from liquidloc.analysis.statistics_runner import build_smoke_only_statistics_payload, build_statistics_payload, build_repeat_summary_rows, build_scene_summary_rows  # 统计编排：repeat/scene 汇总、方法级统计载荷。
_build_repeat_summary_rows = build_repeat_summary_rows
_build_scene_summary_rows = build_scene_summary_rows
from liquidloc.common.constants import (
    GROUND_TRUTH_BY_TASK_ID,
    GROUND_TRUTH_BY_SCENE_VARIANT_ID,
    GROUND_TRUTH_BY_SCENE_ID,
    GROUND_TRUTH_BY_SEQ_ID,
)  # 真值覆盖表键名常量（D9 单源）。
from liquidloc.common.constants import (
    COMPARISON_STATUS_GROUND_TRUTH_BACKED,
    COMPARISON_STATUS_SMOKE_ONLY_SELF_GROUND_TRUTH,
    GT_FILE_NAME,
    LONG_FORM_METRIC_KEYS,
)  # D9：真值文件名、对比状态、长表字段名单源常量，避免字面量在 statistics_runner 与 eval_pipeline 之间漂移。
from liquidloc.common.io_utils import dumps_json_text, read_json  # D10：提升到模块顶层，避免函数内延迟 import 导致重复导入或 NameError 风险；dumps_json_text 与 core_pipeline.py L53 同口径顶层导入。
from liquidloc.common.paths import get_standard_dirs, resolve_output_root  # 解析项目标准目录结构和统一输出根目录。
from liquidloc.common.types import MetricRow, StageResult  # 使用统一的指标行和阶段结果类型。
from liquidloc.common.validation import coerce_finite_scalar, is_numeric, is_string_like, normalize_optional_string, validate_path_component  # 统一判断数值类型、路径校验和有限性校验。
from liquidloc.factories.metric_factory import create_metric_calculator  # 按指标名创建具体计算器。
from liquidloc.interfaces.pipeline_api import PipelineAPI, normalize_pipeline_cfg  # 继承统一流水线接口，并复用统一配置规整 helper。
from liquidloc.metrics.metric_runner import compute_metrics  # 计算 bundle 对应的核心指标。
from liquidloc.protocol.experiment_gates import load_experiment_protocol, normalize_eval_request, get_seed_policy  # 读取并校验实验协议。seed 量级门用 get_seed_policy 读取 n_seed_min。
from liquidloc.protocol.scene_axis_protocol import SCENE_AXES as _SCENE_AXES  # 从协议层导入冻结轴名，避免硬编码漂移。
from liquidloc.protocol.metric_schema import get_metric_meta, get_metric_order  # 读取指标元信息和协议顺序。
from liquidloc.protocol.scene_schema import decode_scene  # D10：提升到模块顶层，避免函数内延迟 import 导致重复导入和 NameError 风险。


_PLOTTING_AXES = _SCENE_AXES  # 绘图和 sweep 固定使用的轴顺序。
_LONG_FORM_KEYS = LONG_FORM_METRIC_KEYS  # D9 单源：引用 common/constants.py 的冻结集合，禁止本地重复定义。指标长表专用 5 字段（metric/value/unit/direction/group）。
_SUPPORT_FIELDS = (  # 指标长表里附带的支撑字段。
    'prediction_length',  # 预测轨迹长度。
    'ground_truth_length',  # 真值轨迹长度。
    'aligned_length',  # 成功对齐后的长度。
    'valid_pair_count',  # 有效配对点数量。
    'overlap_ratio',  # 预测与真值的重叠比例。
    'reliability_status',  # 结果可靠性状态。
)
_RUNTIME_METRIC_NAMES = tuple(  # 只收 runtime 组指标名。
    metric_name  # 只保留指标名本身。
    for metric_name, meta in get_metric_meta().items()  # 遍历所有指标元信息。
    if meta.get('group') == 'runtime'  # 只筛 runtime 组的指标名。
)


def _load_ground_truth_map(ground_truth_root, seq_ids: list[str]) -> dict[str, list[dict[str, Any]]]:  # 按序列编号把真值文件读成内存映射。
    """按序列编号从真值根目录加载 `gt.json` 映射。

    Args:  # 参数说明开始。
        ground_truth_root: 真值根目录，里面通常按 `seq_id/gt.json` 组织。
        seq_ids: 需要读取的序列编号列表。

    Returns:  # 返回值说明开始。
        `seq_id -> 真值行列表` 的映射；没有根目录时返回空字典。
    """
    if not ground_truth_root:  # 没有真值根目录时，不需要继续找文件。
        return {}  # 直接返回空表，表示这批评估没有可加载的外部真值。
    root = Path(ground_truth_root)  # 把根路径统一转成 Path，后面拼接更稳。
    gt_map: dict[str, list[dict[str, Any]]] = {}  # 每个 seq_id 对应一组真值行。
    for seq_id in seq_ids:  # 逐个序列去找对应的真值文件。
        validate_path_component(seq_id, name='seq_id')  # 校验 seq_id 不含路径穿越字符。
        gt_path = root / seq_id / GT_FILE_NAME  # 约定的真值文件路径（D9：使用单源常量）。
        if gt_path.is_file():  # 只有文件真的存在才读，避免空路径报错。
            raw_gt = read_json(gt_path)  # 读出并解析 JSON（D10：已提升到模块顶层）。
            if not isinstance(raw_gt, list):  # D5：确保返回值是列表类型。
                raise TypeError(f"{gt_path} must contain a list payload, got {type(raw_gt).__name__}")
            validated_gt: list[dict[str, Any]] = []  # 校验后的真值行列表。
            for row_index, row in enumerate(raw_gt):  # D5：逐行校验数值有限性。
                if not isinstance(row, Mapping):  # 每行必须是映射类型。
                    raise TypeError(f"{gt_path} row {row_index} must be a dict, got {type(row).__name__}")
                validated_row: dict[str, Any] = {}  # 当前行的校验后副本。
                for key, value in row.items():  # D5：对每个字段做有限性校验。
                    if is_numeric(value):  # 仅对数值类型做有限性检查。
                        # D5：coerce_finite_scalar 拒绝 NaN/Inf，防止超大数字绕过 parse_constant。
                        validated_row[key] = coerce_finite_scalar(value, name=f"{gt_path} row {row_index} key '{key}'")
                    else:
                        validated_row[key] = value  # 非数值字段直接保留。
                validated_gt.append(validated_row)  # 校验通过的行加入结果列表。
            gt_map[seq_id] = deepcopy(validated_gt)  # D3：深拷贝隔离嵌套引用，防止跨 case 数据漂移。
    return gt_map  # 返回 seq_id -> 真值列表 的映射。


def _copy_ground_truth_records(records: Any) -> list[dict[str, Any]] | None:  # 把真值记录转成可安全修改的普通字典列表。
    """把真值记录复制成普通字典列表，保证后续不会改到原始对象。

    Args:  # 参数说明开始。
        records: 可能是列表、可迭代对象或 `None` 的真值记录容器。

    Returns:  # 返回值说明开始。
        新的字典列表；如果输入是 `None`，则返回 `None`。
    """
    if records is None:  # 没有记录时要保留 None 语义。
        return None  # 缺失时保持 None 语义。
    if not isinstance(records, list):  # 允许传入任意可迭代对象。
        records = list(records)  # 先 materialize 成列表，避免后面只读一次就耗尽。
    copied_records: list[dict[str, Any]] = []  # 返回值用独立列表承载。
    for index, row in enumerate(records):  # 逐行检查并复制，index 用于报错定位。
        if not isinstance(row, Mapping):  # 每一行都必须像字典，方便后续按 key 读取。
            raise TypeError(f'ground-truth rows must be mappings, got {type(row).__name__} at index {index}')  # 不合规就直接报错。
        copied = deepcopy(row)  # 深拷贝，隔离嵌套结构（如 cov 矩阵）的引用。
        for key in ('t', 'x', 'y', 'z'):  # 校验标量数值字段的有限性。
            if key in copied and is_numeric(copied[key]):
                copied[key] = coerce_finite_scalar(copied[key], name=f'ground_truth_{key}[{index}]')
        if 'cov' in copied and isinstance(copied['cov'], Sequence) and not isinstance(copied['cov'], (str, bytes)):  # 校验协方差矩阵的有限性。
            for i, cov_row in enumerate(copied['cov']):
                if isinstance(cov_row, Sequence) and not isinstance(cov_row, (str, bytes)):
                    for j, val in enumerate(cov_row):
                        if is_numeric(val):
                            copied['cov'][i][j] = coerce_finite_scalar(val, name=f'ground_truth_cov[{index}][{i}][{j}]')
        copied_records.append(copied)  # 追加经过深拷贝和有限性校验的记录。
    return copied_records  # 返回可以安全修改的新列表。


def _resolve_ground_truth_records(  # 按任务覆盖规则解析某个 bundle 对应的真值记录。
    bundle: Mapping[str, Any],  # 当前预测 bundle，里面可能带 task_id、scene_id、seq_id 等字段。
    cfg: Mapping[str, Any],  # 评估配置，里面可能显式提供真值覆盖表。
    gt_map: Mapping[str, list[dict[str, Any]]],  # 从真值根目录加载出的 seq_id -> 真值行列表映射。
) -> list[dict[str, Any]] | None:  # 找到的真值记录列表；如果没有任何可用来源，则返回 None。
    """按任务覆盖规则解析某个 bundle 对应的真值记录。

    Args:  # 参数说明开始。
        bundle: 当前预测 bundle，里面可能带 `task_id`、`scene_id`、`seq_id` 等字段。
        cfg: 评估配置，里面可能显式提供真值覆盖表。
        gt_map: 从真值根目录加载出的 `seq_id -> 真值行列表` 映射。

    Returns:  # 返回值说明开始。
        找到的真值记录列表；如果没有任何可用来源，则返回 `None`。
    """
    override_candidates = (  # 覆盖顺序从最细粒度到最粗粒度，避免较粗键误命中。
        (GROUND_TRUTH_BY_TASK_ID, normalize_optional_string(bundle.get('task_id'))),  # 任务级覆盖最精确。
        (GROUND_TRUTH_BY_SCENE_VARIANT_ID, normalize_optional_string(bundle.get('scene_variant_id'))),  # 场景变体级次之。
        (GROUND_TRUTH_BY_SCENE_ID, normalize_optional_string(bundle.get('scene_id'))),  # 场景级再往后退一档。
        (GROUND_TRUTH_BY_SEQ_ID, normalize_optional_string(bundle.get('seq_id'))),  # 序列级是最后的显式覆盖。
    )
    for mapping_name, key in override_candidates:  # 逐层尝试不同粒度的覆盖表。
        override_map = cfg.get(mapping_name)  # 逐层读取配置中的覆盖表。
        if not isinstance(override_map, Mapping) or not key:  # 缺映射表或缺键时直接跳过。
            continue  # 这层没有可用命中条件，就看下一层。
        if key in override_map:  # 只要命中就直接用覆盖表，不再往后回退。
            return _copy_ground_truth_records(override_map[key])  # 命中覆盖项后直接返回复制结果。

    seq_id = normalize_optional_string(bundle.get('seq_id'))  # 再回退到序列级真值映射。
    if not seq_id:  # 连序列编号都没有就无法定位外部真值。
        return None  # 连 seq_id 都没有就无法定位真值。
    records = gt_map.get(seq_id)
    if not isinstance(records, list):  # 类型守卫：与 override_map 处理模式一致。
        return None
    return _copy_ground_truth_records(records)  # 再按序列级映射读默认真值。


def _parse_scene_axes(scene_id: str) -> dict[str, str]:  # 把 S(...) 形式的场景编号拆成轴级字典。
    """从 scene_id 解析各轴取值，返回轴名到轴值的映射。

    Args:  # 参数说明开始。
        scene_id: 形如 `S(A3,N3,V2,K1)` 的场景编号字符串（按五轴档位协议：G 已合并入 K）。

    Returns:  # 返回值说明开始。
        轴名到轴级标签的字典；格式不对时返回空字典。
    """
    try:
        spec = decode_scene(scene_id)
        return {_SCENE_AXES[0]: spec.A_level, _SCENE_AXES[1]: spec.N_level, _SCENE_AXES[2]: spec.V_level, _SCENE_AXES[3]: spec.K_value, _SCENE_AXES[4]: spec.M_level}
    except (ValueError, TypeError):
        return {}


def _resolve_bundle_axes(bundle: Mapping[str, Any]) -> dict[str, str]:  # 合并 bundle 里的 scene_id 和显式 axes 字段。
    """合并 bundle 里的 scene_id 和显式 axes 字段。

    Args:  # 参数说明开始。
        bundle: 预测 bundle 或案例记录，可能同时带 `scene_id` 和 `axes`。

    Returns:  # 返回值说明开始。
        最终合并后的轴级字典。
    """
    resolved_axes = _parse_scene_axes(str(bundle.get('scene_id', '')))
    raw_axes = bundle.get('axes')  # 显式轴字段优先补充到解析结果里。
    if isinstance(raw_axes, Mapping) and raw_axes:  # 显式轴字段存在且非空时才合并。
        for axis, level in raw_axes.items():  # 逐个轴级项写回到解析结果里。
            if not is_string_like(axis) or not (is_string_like(level) or is_numeric(level)):  # 键值类型不对时直接跳过。
                continue  # 非法键值不参与合并。
            if is_numeric(level):  # D5：数值型 level 必须有限，避免 NaN/inf 被转成 'nan'/'inf' 垃圾标签。
                try:
                    if not math.isfinite(float(level)):  # 非有限数值跳过。
                        continue  # NaN/inf 不参与合并。
                except (OverflowError, TypeError, ValueError):  # 大整数或不可转换数值跳过。
                    continue  # 不可转换的数值不参与合并。
            normalized_axis = str(axis).strip()  # 轴名去空白后再处理。
            if not normalized_axis:  # 空轴名没有意义。
                continue  # 空轴名跳过。
            if normalized_axis not in _SCENE_AXES:  # D9：只接受协议冻结的轴名，防止非协议轴漂移。
                continue  # 非协议轴名跳过。
            normalized_level = str(level).strip()  # 轴级统一转成字符串。
            if not normalized_level:  # 空轴级也没有意义。
                continue  # 空级别跳过。
            if not normalized_level.startswith(normalized_axis):  # 如果没带轴名前缀，就补完整标签。
                normalized_level = f'{normalized_axis}{normalized_level}'  # 补成完整轴级标签。
            resolved_axes[normalized_axis] = normalized_level  # 覆盖或补充到最终结果。
    return resolved_axes


def _validate_visual_protocol_consistency(bundle: Mapping[str, Any]) -> None:  # 检查视觉轴报告里的协议一致性标志是否自洽。
    """检查视觉轴报告里的协议一致性标志是否自洽。"""
    scenario_context = bundle.get('scenario_context')  # 场景上下文里才会带各轴报告。
    if not isinstance(scenario_context, Mapping):  # 没有场景上下文时没法检查。
        return  # 没有场景上下文就不做一致性检查。
    scenario_reports = scenario_context.get('scenario_reports')  # 场景上下文里的各轴报告集合。
    if not isinstance(scenario_reports, Mapping):  # 没有报告就不检查。
        return  # 没有报告就不检查。
    visual_report = scenario_reports.get(_SCENE_AXES[2])  # D9 单源：视觉轴名引用 _SCENE_AXES 常量，与 _parse_scene_axes 保持一致，避免字面量漂移。
    if not isinstance(visual_report, Mapping):  # 不是映射型报告就没法继续读字段。
        return  # 不是映射型报告就跳过。
    consistency_checks = visual_report.get('consistency_checks')  # 视觉轴自己的逐项检查结果。
    if isinstance(consistency_checks, Mapping) and consistency_checks:  # 有逐项检查结果时优先用它。
        resolved_protocol_consistent = all(bool(flag) for flag in consistency_checks.values())  # 由所有检查项共同决定。
        declared_protocol_consistent = bool(visual_report.get('protocol_consistent', True))  # 报告里显式声明的结论。
        if declared_protocol_consistent != resolved_protocol_consistent:  # 声明和逐项检查不一致时要报错。
            raise ValueError(  # 声明和逐项检查不一致时要报错。
                f"prediction bundle {bundle.get('task_id') or bundle.get('scene_id') or '<unknown>'} "
                'carries a V-axis report whose protocol_consistent conflicts with consistency_checks'
            )
        if not resolved_protocol_consistent:  # 逐项检查失败时也要报错。
            raise ValueError(  # 逐项检查失败时也要报错。
                f"prediction bundle {bundle.get('task_id') or bundle.get('scene_id') or '<unknown>'} "
                "carries a V-axis report with protocol_consistent=False"
            )
        return  # 已经通过一致性检查时提前返回。
    if visual_report.get('protocol_consistent', True) is False:  # 没有逐项检查时，显式 False 也要报错。
        raise ValueError(  # 没有逐项检查时，显式 False 也要报错。
            f"prediction bundle {bundle.get('task_id') or bundle.get('scene_id') or '<unknown>'} "
            "carries a V-axis report with protocol_consistent=False"
        )


def _build_metric_rows(metric_values: Mapping[str, Any]) -> list[MetricRow]:  # 把指标字典按既定顺序组装成 MetricRow 列表。
    """把指标字典按既定顺序组装成 MetricRow 列表。"""
    metric_order = get_metric_order()  # 先拿协议里规定的指标顺序，后面必须严格对齐。
    metric_names = list(metric_values.keys())  # 先检查输入是否已经按约定顺序排列。
    if metric_names != metric_order:  # 顺序不一致会破坏长表和统计口径。
        raise ValueError(  # 同时展示期望顺序和实际顺序，便于定位上游漂移点。
            f'metric values must be ordered exactly as get_metric_order(): '
            f'expected {metric_order}, got {metric_names}'
        )
    return [create_metric_calculator(metric_name, {}).compute(metric_values[metric_name]) for metric_name in metric_order]  # 按协议顺序逐项生成 MetricRow。


def _extract_metric_name_value(metric_row: Any) -> tuple[str, Any]:  # 从 dict 或对象型 metric row 中提取名称和值。
    """从 dict 或对象型 metric row 中提取名称和值。"""
    if isinstance(metric_row, Mapping):  # 映射型记录直接按键读取。
        metric_name = metric_row.get('metric')  # 映射型记录直接取键。
        metric_value = metric_row.get('value')
    else:  # 对象型记录就取属性。
        metric_name = getattr(metric_row, 'metric', None)  # 对象型记录就取属性。
        metric_value = getattr(metric_row, 'value', None)

    if not is_string_like(metric_name) or not str(metric_name).strip():  # 名称必须是非空字符串。
        raise ValueError(f'metric rows must expose a non-empty metric name, got {metric_name!r}')  # D8：值合同违例统一抛 ValueError，与 statistics_runner/types 对齐。
    return str(metric_name).strip(), metric_value  # D3：返回规范化指标名（剥离空白并转 Python str）和值，供后续写入表格。


def _build_case_ref(bundle: dict[str, Any]) -> str:  # 生成一个可稳定分组的 case_ref。
    """生成一个可稳定分组的 case_ref。

    分组键构造规则（确定性优先级，同 bundle 多次调用结果一致）：
      1. 有 task_id 时走任务级 case_ref：`{task_id}::{method_name}[::{scene_id}][::{repeat_id}]`，
         scene_id 与 repeat_id 仅在非空时条件性拼入（baseline 等无 scene_id 的 case 不会硬塞空段）；
         task_id 路径下 method_name 缺失则 raise ValueError。
      2. 无 task_id 时回退到字段组合：按 scene_id、seq_id、method_name、repeat_id 顺序收集非空值，
         用 `::` 拼接；至少能拼出一个字段则返回，否则 raise ValueError。
    """
    raw_task_id = normalize_optional_string(bundle.get('task_id'))
    raw_method_name = normalize_optional_string(bundle.get('method_name'))
    raw_repeat_id = normalize_optional_string(bundle.get('repeat_id'))
    raw_scene_id = normalize_optional_string(bundle.get('scene_id'))
    if raw_task_id:  # 有 task_id 时走任务级 case_ref。
        if not raw_method_name:  # 没有 method_name 就无法拼出任务级稳定键。
            raise ValueError('prediction bundle is missing method_name and cannot derive a task-based case_ref')
        case_ref = f'{raw_task_id}::{raw_method_name}'
        if raw_scene_id:  # 把 scene_id 拼入 case_ref，区分同一 task 下不同场景变体。
            case_ref = f'{case_ref}::{raw_scene_id}'
        if raw_repeat_id:  # 有重复编号时把它也拼进分组键。
            return f'{case_ref}::{raw_repeat_id}'
        return case_ref  # 没有重复编号就返回任务+方法+场景的三段键。

    ref_parts = []  # 没有 task_id 时就退回用更宽的字段组合。
    for field_name in ('scene_id', 'seq_id', 'method_name', 'repeat_id'):  # 按稳定性从高到低收集备用键。
        raw_value = normalize_optional_string(bundle.get(field_name))
        if raw_value:  # 只有有效字段值才纳入组合。
            ref_parts.append(raw_value)
    if ref_parts:  # 只要至少拼出一个字段，就用这些字段组成回退键。
        return '::'.join(ref_parts)
    raise ValueError('prediction bundle is missing task_id and all fallback fields (scene_id/seq_id/method_name/repeat_id) are empty, cannot derive a case_ref')


def _build_case_record(bundle: dict[str, Any], rows, *, case_ref: str) -> dict[str, Any]:  # 把一个 bundle 和它的指标行拼成面向表格消费的 case 记录。
    """把一个 bundle 和它的指标行拼成面向表格消费的 case 记录。

    Args:  # 参数说明开始。
        bundle: 当前预测 bundle，里面带着 seq_id、scene_id、method_name 等身份字段。
        rows: 这个 bundle 对应的一组 MetricRow 或类似记录。
        case_ref: 已经算好的稳定分组键，直接写入记录里。

    Returns:  # 返回值说明开始。
        可以直接写入 case 表或 sweep 表的扁平记录。
    """
    record = {  # 先构造 case 级记录，再补可选字段。
        'case_ref': case_ref,  # 稳定分组键，供所有下游表对齐。
        'seq_id': bundle['seq_id'],  # 序列编号，标识这个 case 属于哪条数据序列。
        'scene_id': bundle['scene_id'],  # 场景编号，标识几何和环境上下文。
        'method_name': bundle['method_name'],  # 方法名，标识这个 case 属于哪个算法分支。
    }
    task_id = normalize_optional_string(bundle.get('task_id'))  # 任务编号如果存在就规整后写入。
    if task_id:  # 有 task_id 才补进去，避免空字符串污染表格。
        record['task_id'] = task_id  # 记录任务编号，方便人工回查。
    repeat_id = normalize_optional_string(bundle.get('repeat_id'))  # 重复编号同样先规整。
    if repeat_id:  # repeat_id 同理，只在有效时写入。
        record['repeat_id'] = repeat_id  # 记录重复编号，方便做多次实验区分。

    if rows:  # 如果有指标行，就把第一行里的补充字段也带上。
        for key, value in rows[0].items():  # 只从第一行抽一次元信息，避免重复覆盖。
            if key not in _LONG_FORM_KEYS and key not in record:  # 长表专用字段和已存在字段都不再重复写入。
                record[key] = deepcopy(value)  # D3：深拷贝隔离，避免与 rows[0] 共享可变引用导致跨 case 数据漂移。

    for row in rows:  # 再把每个 MetricRow 的核心指标值塞进记录。
        metric_name, metric_value = _extract_metric_name_value(row)  # 把当前指标行拆成名字和值。
        record[metric_name] = deepcopy(metric_value)  # D3：深拷贝隔离，避免与 MetricRow 共享可变引用导致跨 case 数据漂移。
    return record  # 返回面向表格消费的单条 case 记录。


def _build_sweep_row(bundle: dict[str, Any], rows, *, case_ref: str) -> dict[str, Any]:  # 把 case 记录再补成适合 sweep 表的一行。
    """把 case 记录再补成适合 sweep 表的一行。

    Args:  # 参数说明开始。
        bundle: 当前预测 bundle。
        rows: 这个 case 的指标行。
        case_ref: 稳定分组键。

    Returns:  # 返回值说明开始。
        可直接写入 sweep 表的一行记录。
    """
    record = _build_case_record(bundle, rows, case_ref=case_ref)  # 先复用 case 记录，再补上轴字段。
    axes = _resolve_bundle_axes(bundle)  # 解析出用于 sweep 的轴级标签。
    for axis_name in _PLOTTING_AXES:  # 按固定顺序把各轴写回行里。
        if axis_name in axes:  # 只有确实存在这个轴时才写入。
            record[axis_name] = axes[axis_name]  # 逐轴写回，保持 sweep 列稳定。
    return record  # 返回可直接写 sweep 表的一行。


def _build_gt_bundle_record(bundle: dict[str, Any], gt_records: list[dict[str, Any]], *, case_ref: str) -> dict[str, Any]:  # 把某个案例的真值记录整理成单独 bundle。
    """把某个案例的真值记录整理成单独 bundle。

    Args:  # 参数说明开始。
        bundle: 预测 bundle，提供 case 的身份字段。
        gt_records: 与该 case 对应的真值记录列表。
        case_ref: 稳定分组键。

    Returns:  # 返回值说明开始。
        一个单独的真值 bundle 字典。
    """
    gt_bundle = {  # 先把真值 bundle 的主字段装起来。
        'case_ref': case_ref,  # 真值 bundle 也要带稳定分组键。
        'seq_id': bundle['seq_id'],  # 序列编号。
        'scene_id': bundle['scene_id'],  # 场景编号。
        'method_name': bundle['method_name'],  # 方法名。
        'states': deepcopy(gt_records),  # D3：深拷贝隔离嵌套引用，本函数对同一 gt_records 被调用两次，浅拷贝会导致 gt_bundle/trajectory_gt_bundle 共享 dict。
    }
    task_id = normalize_optional_string(bundle.get('task_id'))  # 有 task_id 就一并保留。
    if task_id:  # 真值 bundle 也保留 task_id，方便回查来源。
        gt_bundle['task_id'] = task_id  # 写入任务编号。
    repeat_id = normalize_optional_string(bundle.get('repeat_id'))  # 重复编号也要保留。
    if repeat_id:  # 重复编号同样要保留。
        gt_bundle['repeat_id'] = repeat_id  # 写入重复编号。
    return gt_bundle  # 返回单独整理好的真值 bundle。


def _build_trajectory_bundle_record(bundle: dict[str, Any], *, case_ref: str) -> dict[str, Any]:  # 把预测轨迹整理成适合下游画图的 bundle。
    """把预测轨迹整理成适合下游画图的 bundle。

    Args:  # 参数说明开始。
        bundle: 预测 bundle，里面必须带 `states`。
        case_ref: 当前 case 的稳定分组键。

    Returns:  # 返回值说明开始。
        适合轨迹图和对比图消费的 bundle。
    """
    trajectory_bundle = {  # 先把预测轨迹 bundle 的主字段装起来。
        'case_ref': case_ref,  # 稳定分组键。
        'seq_id': bundle['seq_id'],  # 序列编号。
        'scene_id': bundle['scene_id'],  # 场景编号。
        'method_name': bundle['method_name'],  # 方法名。
        'states': deepcopy(bundle['states']),  # D3：深拷贝隔离嵌套引用，state 内 cov 矩阵/position list 等嵌套可变结构若浅拷贝会与 bundle['states'] 共享引用，被下游 plotting/analysis 写回时反向污染上游 bundles 入参；与 L387 _build_gt_bundle_record 同口径，deepcopy 已在 L16 导入。
    }
    task_id = normalize_optional_string(bundle.get('task_id'))  # 有任务编号时也保留。
    if task_id:  # 预测轨迹 bundle 也带上 task_id。
        trajectory_bundle['task_id'] = task_id  # 写入任务编号。
    repeat_id = normalize_optional_string(bundle.get('repeat_id'))  # 重复编号也保留。
    if repeat_id:  # 预测轨迹 bundle 也带上 repeat_id。
        trajectory_bundle['repeat_id'] = repeat_id  # 写入重复编号。
    timestamps = bundle.get('timestamps')  # 轨迹采样时间戳，如果有就一并保留。
    if isinstance(timestamps, Sequence) and not isinstance(timestamps, (str, bytes)) and timestamps:  # 只有真正的非空序列才写入。
        trajectory_bundle['timestamps'] = list(timestamps)  # 写入时间戳副本。
    return trajectory_bundle  # 返回可直接画图的预测轨迹 bundle。


def _compute_n_seed_from_bundles(bundles: list[dict[str, Any]]) -> int:  # §19.2：从 bundles 派生 n_seed（唯一 repeat_id 数）。
    """从预测 bundles 中统计跨种子数（unique repeat_id 数）。

    repeat_id 由 core_pipeline 写入 bundle（task['repeat_id']），
    每个 trajectory seed 对应一个唯一 repeat_id，缺失时退化为 1。
    """
    repeat_ids: set[str] = set()
    for b in bundles:
        rid = b.get('repeat_id')
        if rid is not None:
            repeat_ids.add(str(rid))
    return max(len(repeat_ids), 1)  # 至少为 1，防止空数据误判。


def _is_ablation_method(method_name: str) -> bool:  # §19.3：判断方法名是否为消融变体。
    """消融方法名通常以 _wo_（without）或 _ablation_ 为后缀前缀。"""
    if not method_name:
        return False
    m = str(method_name).lower()
    return '_wo_' in m or '_ablation_' in m or m.startswith('ablation_')


def _build_plotting_inputs(  # 把评估结果整理成绘图消费的标准输入结构。
    bundles: list[dict[str, Any]],  # 预测 bundle 列表，里面包含每个 case 的主结果。
    gt_records_by_case_ref: Mapping[str, list[dict[str, Any]]],  # case_ref -> 真值记录，用于轨迹对比。
    sweep_rows: list[dict[str, Any]],  # 已经整理好的 sweep 表行。
    metric_rows: list[dict[str, Any]],  # 原始长表指标行。
    runtime_table_rows: list[dict[str, Any]],  # 运行时统计表行。
    main_table_rows: list[dict[str, Any]],  # 已聚合主表行。
) -> dict[str, Any]:  # 返回一整套绘图输入字典。
    """把评估结果整理成绘图消费的标准输入结构。"""
    gt_bundle_rows: list[dict[str, Any]] = []  # 真值 bundle 列表，供轨迹图和审计使用。
    trajectory_prediction_rows: list[dict[str, Any]] = []  # 预测轨迹 bundle 列表，供轨迹对比使用。
    trajectory_gt_rows: list[dict[str, Any]] = []  # 真值轨迹 bundle 列表，供轨迹对比使用。
    for bundle in bundles:  # 逐个 bundle 组装轨迹对比所需的两个 bundle 列表。
        case_ref = _build_case_ref(bundle)  # 当前 bundle 的稳定分组键。
        gt_records = gt_records_by_case_ref.get(case_ref)  # 该 case 对应的真值记录。
        if gt_records:  # 只有确实有真值时才生成对应 bundle。
            gt_bundle_rows.append(_build_gt_bundle_record(bundle, gt_records, case_ref=case_ref))  # 真值 bundle 入列。
            trajectory_prediction_rows.append(_build_trajectory_bundle_record(bundle, case_ref=case_ref))  # 预测轨迹 bundle 入列。
            trajectory_gt_rows.append(_build_gt_bundle_record(bundle, gt_records, case_ref=case_ref))  # 真值轨迹 bundle 入列。

    return {  # 返回绘图脚本直接消费的总字典。
        'metric_table': deepcopy(metric_rows),  # D3：深拷贝隔离入参引用，防止下游消费反向污染调用方局部变量。
        'main_table': deepcopy(main_table_rows),  # D3：深拷贝隔离入参引用，与 L101/L122 模式对齐。
        'runtime_table': deepcopy(runtime_table_rows),  # D3：深拷贝隔离入参引用，调用方在 L657 仍直接使用原列表，必须隔离。
        'sweep_table': deepcopy(sweep_rows),  # D3：深拷贝隔离入参引用，防止 sweep 行被下游误改。
        'gt_bundle': gt_bundle_rows,  # 真值 bundle（本函数内部新建列表，无需额外拷贝）。
        'trajectory_bundle': {  # 轨迹 bundle 继续分成预测和真值两部分。
            'prediction_bundle': trajectory_prediction_rows,  # 预测轨迹列表（内部新建，无需拷贝）。
            'gt_bundle': trajectory_gt_rows,  # 对应真值轨迹列表（内部新建，无需拷贝）。
        },  # 结束轨迹 bundle 子结构。
    }  # 结束绘图输入总字典。


def _build_runtime_table(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:  # 从长表 metric_rows 里重建运行时统计表。
    """从长表 metric_rows 里重建运行时统计表。"""
    runtime_rows_by_case: dict[str, dict[str, Any]] = {}  # case_ref -> 聚合后的运行时统计行。
    for row in metric_rows:  # 逐行扫描长表，只抽 runtime 指标。
        metric_name = row.get('metric')  # 当前行的指标名。
        if metric_name not in _RUNTIME_METRIC_NAMES:  # 非 runtime 指标直接跳过。
            continue  # 不是 runtime 指标就跳过。
        case_ref = row.get('case_ref')  # 当前指标所属 case_ref。
        if not is_string_like(case_ref) or not str(case_ref).strip():  # case_ref 必须是非空字符串。
            raise ValueError('runtime metric rows must contain a non-empty case_ref')
        base_row = {key: deepcopy(value) for key, value in row.items() if key not in _LONG_FORM_KEYS}  # D3：深拷贝隔离嵌套引用，防止 metric_rows 中潜在嵌套结构被下游反向污染；与 L101/L122/L413 deepcopy 口径对齐。
        runtime_row = runtime_rows_by_case.setdefault(case_ref, base_row)  # 同 case_ref 共用一行；base_row 已是新建字典且值已深拷贝，无需再 dict() 浅拷贝。
        runtime_row[metric_name] = row.get('value')  # 回填 runtime 指标值（标量 float，不可变，无需深拷贝）。

    runtime_rows = list(runtime_rows_by_case.values())  # 聚合后的运行时表。
    if not runtime_rows:  # 如果一个 runtime 指标都没有，返回空表而非崩溃，允许协议不定义 runtime 指标。
        return []

    for runtime_row in runtime_rows:  # 逐行检查每个 case 是否都含有所有 runtime 指标。
        missing_metrics = [metric_name for metric_name in _RUNTIME_METRIC_NAMES if metric_name not in runtime_row]  # 检查缺失项。
        if missing_metrics:  # 有缺项就立刻报错，避免输出不完整表。
            case_ref = runtime_row.get('case_ref', '<unknown>')  # 报错时标明 case。
            raise ValueError(f"runtime row for {case_ref} is missing runtime metrics: {missing_metrics}")
    return runtime_rows  # 返回聚合后的运行时表。


def _group_metric_rows_by_case(metric_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:  # 按 case_ref 给长表指标分组，并确认每组指标顺序完全一致。
    """按 case_ref 给长表指标分组，并确认每组指标顺序完全一致。"""
    grouped_rows: dict[str, list[dict[str, Any]]] = {}  # case_ref -> 对应的指标长表行列表。
    for row in metric_rows:  # 逐行聚合到对应的 case 分组里。
        case_ref = row.get('case_ref')  # 当前行所属的分组键。
        if not is_string_like(case_ref) or not str(case_ref).strip():  # case_ref 不能为空。
            raise ValueError('metric_table rows must contain a non-empty case_ref')
        grouped_rows.setdefault(case_ref, []).append(dict(row))  # 先浅拷贝再分组，避免后续共享引用被改写。

    metric_order = get_metric_order()  # 分组后再校验每个 case 的指标顺序。
    for case_ref, case_rows in grouped_rows.items():  # 每个 case 分组都要严格校验顺序。
        case_metric_names = [str(case_row.get('metric')) for case_row in case_rows]  # 取出该 case 的实际指标顺序。
        if case_metric_names != metric_order:  # 指标顺序必须完全对齐协议。
            raise ValueError(  # 同时展示期望顺序和实际顺序，便于定位上游漂移点，与 L262-265 同口径。
                f'case {case_ref} must cover get_metric_order() exactly once and in order; '
                f'expected {metric_order}, got {case_metric_names}'
            )
    return grouped_rows


def _build_case_views(  # 把 bundle 和分组指标拼成 case_catalog 与 sweep_rows。
    bundles: list[dict[str, Any]],  # 原始预测 bundle 列表，每个 case 都要从这里找回 bundle。
    grouped_metric_rows: dict[str, list[dict[str, Any]]],  # 已经按 case_ref 分好组的指标长表。
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:  # 返回 case_catalog 和 sweep_rows 两部分。
    """把 bundle 和分组指标拼成 case_catalog 与 sweep_rows。

    Args:  # 参数说明开始。
        bundles: 原始预测 bundle 列表。
        grouped_metric_rows: 按 case_ref 聚合后的指标长表。

    Returns:  # 返回值说明开始。
        `case_catalog` 和 `sweep_rows` 组成的二元组。
    """
    bundle_by_case_ref: dict[str, dict[str, Any]] = {}  # case_ref -> 原始 bundle。
    for bundle in bundles:  # D7/D8：逐个构建映射，遇到重复 case_ref 立即报错，与 case_selector.py L66-69/L90-91 同口径，防止字典推导式静默丢弃重复 bundle 破坏公平性。
        case_ref = _build_case_ref(bundle)
        if case_ref in bundle_by_case_ref:  # 重复 case_ref 会导致字典推导式静默丢弃较早的 bundle，违反 D7 公平性。
            raise ValueError(f'prediction bundles contain duplicate case_ref: {case_ref}')
        bundle_by_case_ref[case_ref] = bundle
    bundle_refs = set(bundle_by_case_ref)  # bundle 侧 case_ref 集合。
    metric_refs = set(grouped_metric_rows)  # 指标侧 case_ref 集合。
    if bundle_refs != metric_refs:  # 两边 case_ref 集合必须完全一致。
        missing_in_bundles = sorted(metric_refs - bundle_refs)  # 指标侧有但 bundle 侧没有的 case_ref。
        missing_in_metrics = sorted(bundle_refs - metric_refs)  # bundle 侧有但指标侧没有的 case_ref。
        raise ValueError(  # D8：报错时展示具体差异，便于定位上游漂移。
            f'prediction bundles and metric_table rows must cover the same case_ref set; '
            f'missing in bundles: {missing_in_bundles}, missing in metrics: {missing_in_metrics}'
        )

    case_catalog: dict[str, dict[str, Any]] = {}  # case_ref -> 面向审计和统计的案例记录。
    sweep_rows: list[dict[str, Any]] = []  # 供 sweep 表使用的扁平行列表。
    for case_ref, case_rows in grouped_metric_rows.items():  # 逐个 case 构造两种视图。
        bundle = bundle_by_case_ref[case_ref]  # 找到这个 case 对应的原始 bundle。
        case_catalog[case_ref] = _build_case_record(bundle, case_rows, case_ref=case_ref)  # 生成 case 级记录。
        sweep_rows.append(_build_sweep_row(bundle, case_rows, case_ref=case_ref))  # 再生成用于 sweep 的记录。
    return case_catalog, sweep_rows  # 按固定顺序返回两个结果。


class EvalPipeline(PipelineAPI):  # 把预测 bundle 转成评估结果、统计表、选例和绘图输入。
    """把预测 bundle 转成评估结果、统计表、选例和绘图输入。"""

    def run(self, pipeline_cfg: dict | None = None, runtime_context: dict | None = None) -> StageResult:  # `runtime_context` 预留给上层编排，当前实现不直接依赖它。
        """执行评估阶段，返回产物路径和元数据。"""
        cfg = normalize_pipeline_cfg(pipeline_cfg)  # 先把输入配置复制成普通字典，并拒绝非映射输入。
        from liquidloc.common.tee_logger import print_dict
        print_dict({
            "prediction_bundles_count": len(cfg.get("prediction_bundles") or []),
            "ground_truth_root": str(cfg.get("ground_truth_root")) if cfg.get("ground_truth_root") else None,
            "output_root": str(cfg.get("output_root")) if cfg.get("output_root") else None,
            "mode": cfg.get("mode"),
            "experiment_protocol_path": str(cfg.get("experiment_protocol_path")) if cfg.get("experiment_protocol_path") else None,
        }, "EvalPipeline.run 入口参数")
        bundles = list(cfg.get('prediction_bundles') or [])  # 这里就是评估的唯一核心输入，里面装着所有预测 bundle。
        if not bundles:  # 没有 bundle 就无法评估。
            raise ValueError('prediction_bundles must be a non-empty list')  # 没有 bundle 就无法评估。

        protocol_cfg = load_experiment_protocol(cfg.get('experiment_protocol_path'))  # 读取实验协议，后面的门槛和优先级都从这里来。
        gate_report = normalize_eval_request(cfg, protocol_cfg)  # 先做协议门控，确认这批评估请求允许继续往下跑。
        normalized_cfg = dict(cfg)  # shallow copy: only top-level keys are mutated below, no nested writes
        normalized_cfg['ground_truth_root'] = gate_report.get('ground_truth_root')

        dirs = get_standard_dirs(cfg.get('project_root'))  # 解析项目标准目录，后面所有落盘路径都基于它拼出来。
        output_root = resolve_output_root(cfg, 'eval_pipeline')  # 统一解析输出根目录（含空白字符串校验）。
        metrics_dir = output_root / 'metrics'  # 指标 CSV 的输出目录。
        statistics_dir = output_root / 'statistics'  # 统计 JSON 的输出目录。
        cases_dir = output_root / 'cases'  # 选例结果的输出目录。
        audits_dir = output_root / 'audits'  # 审计报告的输出目录。
        plotting_dir = output_root / 'plotting_inputs'  # 给后续绘图消费的中间输入目录。
        metrics_dir.mkdir(parents=True, exist_ok=True)  # 逐个确保目录存在，避免后面写文件时报错。
        statistics_dir.mkdir(parents=True, exist_ok=True)  # 统计目录同样要提前建好。
        cases_dir.mkdir(parents=True, exist_ok=True)  # 选例目录也要提前准备。
        audits_dir.mkdir(parents=True, exist_ok=True)  # 审计目录单独存放可追踪结果。
        plotting_dir.mkdir(parents=True, exist_ok=True)  # 绘图输入目录用于给下游可视化脚本直接读取。

        seq_id_set: set[str] = set()  # 收集所有 bundle 的序列编号，后面一次性加载真值。
        for _bundle in bundles:  # D8：先逐 bundle 安全提取 seq_id，缺失或非字符串时抛 ValueError 而非 KeyError。
            _seq_id = normalize_optional_string(_bundle.get('seq_id'))
            if not _seq_id:
                raise ValueError('prediction bundle is missing seq_id')  # D8：值合同违例统一抛 ValueError，与 _build_case_ref L299/L314 同口径。
            seq_id_set.add(_seq_id)
        seq_ids = sorted(seq_id_set)  # 排序保证跨进程可复现。
        gt_map = _load_ground_truth_map(normalized_cfg.get('ground_truth_root'), seq_ids)  # 按序列编号读取真值映射表。
        failure_threshold = coerce_finite_scalar(  # D5：用 coerce_finite_scalar 替代 float()，拒绝 NaN/Inf 经 gate_report 漂移后静默泄漏到 compute_metrics 的 failure_threshold 口径；与 core_pipeline.py L446/L474/L514/L582 同口径。
            gate_report['failure_threshold'],
            name='failure_threshold',
            min_value=0.0,
            inclusive=False,  # 阈值必须严格大于 0，与 metric_runner.compute_metrics L918 `failure_threshold <= 0.0` 校验对齐。
        )
        smoke_without_gt = bool(gate_report.get('smoke_mode')) and not normalized_cfg.get('ground_truth_root')  # 只消费 gate 归一化后的显式布尔 smoke_mode。

        metric_rows: list[dict[str, Any]] = []  # 所有 bundle 展平后的长表指标都收在这里。
        gt_records_by_case_ref: dict[str, list[dict[str, Any]]] = {}  # case_ref -> 真值记录，用于后面拼轨迹和绘图输入。
        failure_segments_by_case_ref: dict[str, list[list[int]]] = {}  # case_ref -> long_failure_segments，按协议 retain_and_audit 要求保留失败帧位置供审计。
        # §9.3 pulse/async 量级门跨单轨违规聚合 (2026-07-23 §9 穷举审视 Round 4 真修复):
        # compute_metrics 在 support_report['section9_pulse_async_violations'] / ['section9_pulse_async_aggregation']
        # 中按 single bundle 生成 §9 pulse/async 违规信息, 但 eval_pipeline 原本只读 _SUPPORT_FIELDS 白名单
        # 内的 6 个基础字段, 静默丢弃这两个 §9 字段 — 导致下游 statistics_table.json / 12_run_statistics
        # 根本看不到单轨 §9 pulse/async 量级门违规, 无法形成"放松则伤" 证据链. 在此聚合:
        # 每 bundle 从 support_report 直接抽 violations + aggregation, 累加到全 method/seq/case 维度,
        # 后写入 audit_payload['section9_pulse_async_audit'] + 落盘 audits_dir/section9_pulse_async_audit.json.
        section9_pulse_async_audit: dict[str, Any] = {
            'violations_total': 0,  # 跨 bundle 累计违规条目数
            'violations_by_method': {},  # {method_name: int}
            'violations_by_seq': {},     # {seq_id: int}
            'violations_by_case': {},    # {case_ref: int}
            'pulse_violated_count': 0,  # 仅 pulse_violated=True 的条目数
            'async_violated_count': 0,  # 仅 async_violated=True 的条目数
            'cmp1_cmp5_at_risk_count': 0,  # cmp1_cmp5_at_risk=True 的条目数
            'violations_detail': [],    # 完整违规条目副本 (供 audits_dir/section9_pulse_async_audit.json 落盘)
            'aggregation_by_bundle': [],  # 每 bundle 的 section9_pulse_async_aggregation 快照 (n_pulse/n_async/n_seq_iter)
            'bundle_count': 0,  # 处理过的 bundle 总数 (含无违规)
        }
        # 偏懒#9 真修复 (2026-07-23 audit Round 1+ 偷懒审视 Round 3): fusion_runner 把 joint_fallback_count
        # 写入 bundle["diagnostics"] 顶层, 但 eval_pipeline 原本不读 diagnostics, 导致 candidate_report 聚合层
        # 完全看不到 fallback count — 文档说"后续 candidate_report 聚合可见"是幻象修复. 在此真聚合:
        # 每 bundle 读 diagnostics.joint_fallback_count + last_joint_fallback_reason + step_joint_attempted,
        # 累加到全 method 维度, 写入 StageResult.metadata.diagnostics_summary, 让 _collect_search_metrics 读到.
        diagnostics_summary: dict[str, Any] = {
            'joint_fallback_count_total': 0,
            'joint_fallback_count_by_method': {},  # {method_name: count}
            'joint_fallback_count_by_seq': {},     # {seq_id: count}
            'last_joint_fallback_reasons_by_method': {},  # {method_name: [reason]}
            'step_joint_attempted_total': 0,
            'step_joint_succeeded_total': 0,
            'bundle_count': 0,
        }
        for bundle in bundles:  # 逐个 bundle 计算指标和长表记录。
            _validate_visual_protocol_consistency(bundle)  # 先检查 bundle 里的视觉轴协议声明是否自洽。
            case_ref = _build_case_ref(bundle)  # 给这个 bundle 生成稳定分组键，后面所有表都靠它对齐。
            # D8：安全提取 bundle 身份字段，缺失时抛 ValueError 而非 KeyError，与 _build_case_ref L299/L314 同口径。
            bundle_seq_id = normalize_optional_string(bundle.get('seq_id'))
            if not bundle_seq_id:
                raise ValueError('prediction bundle is missing seq_id')
            bundle_scene_id = normalize_optional_string(bundle.get('scene_id'))
            if not bundle_scene_id:
                raise ValueError('prediction bundle is missing scene_id')
            bundle_method_name = normalize_optional_string(bundle.get('method_name'))
            if not bundle_method_name:
                raise ValueError('prediction bundle is missing method_name')
            bundle_task_id = normalize_optional_string(bundle.get('task_id'))  # task_id 可选，规整后按需写入，与 repeat_id 同口径。
            # 偏懒#9 真修复 (Round 3): 在此聚合 bundle.diagnostics.joint_fallback_count,
            # 让 StageResult.metadata.diagnostics_summary 真可见, candidate_report / scoring_runs 聚合层也能读到.
            bundle_diagnostics = bundle.get('diagnostics') or {}
            jf_count = int(bundle_diagnostics.get('joint_fallback_count') or 0)
            jf_reason = bundle_diagnostics.get('last_joint_fallback_reason') or ''
            step_joint_attempted = bool(bundle_diagnostics.get('step_joint_attempted'))
            step_joint_succeeded = bool(bundle_diagnostics.get('step_joint_succeeded'))
            diagnostics_summary['joint_fallback_count_total'] += jf_count
            diagnostics_summary['joint_fallback_count_by_method'][bundle_method_name] = (
                diagnostics_summary['joint_fallback_count_by_method'].get(bundle_method_name, 0) + jf_count
            )
            diagnostics_summary['joint_fallback_count_by_seq'][bundle_seq_id] = (
                diagnostics_summary['joint_fallback_count_by_seq'].get(bundle_seq_id, 0) + jf_count
            )
            if jf_reason:
                diagnostics_summary['last_joint_fallback_reasons_by_method'].setdefault(
                    bundle_method_name, []
                ).append(jf_reason)
            if step_joint_attempted:
                diagnostics_summary['step_joint_attempted_total'] += 1
                if step_joint_succeeded:
                    diagnostics_summary['step_joint_succeeded_total'] += 1
            diagnostics_summary['bundle_count'] += 1
            gt_records = _resolve_ground_truth_records(bundle, normalized_cfg, gt_map)  # 解析这个 bundle 对应的真值记录。
            if not gt_records and smoke_without_gt:  # smoke 模式没真值时允许用预测状态临时代替。
                gt_records = _copy_ground_truth_records(bundle.get('states'))  # smoke 场景没有真值时，用预测状态临时代替，保证链路能跑通。
            if gt_records:  # 有真值就写入 case 对照表。
                gt_records_by_case_ref[case_ref] = gt_records  # 真值按 case_ref 存起来，后面生成轨迹 bundle 要用。
            metric_values, support_report = compute_metrics(  # 计算当前 bundle 的主指标和支持字段。
                bundle,  # 当前预测 bundle，是主输入。
                {'states': deepcopy(gt_records or [])},  # D3：深拷贝隔离嵌套引用，gt_records 内部字典（含 cov 矩阵等嵌套可变结构）与 gt_records_by_case_ref[case_ref] 共享引用，compute_metrics 内部 _collect_sequence_data/_extract_traj 若改写会反向污染轨迹 bundle；与 _build_gt_bundle_record L387 deepcopy 同口径。
                failure_threshold=failure_threshold,  # 失败阈值直接来自协议门控。
                return_support=True,  # 明确要求把支持字段也一并返回。
                protocol_cfg=protocol_cfg,  # §9.1 冷启动偏移统一划除 + §9.3 脉冲/异步量级门需读 scene_scale / seed_policy 字段. cold-start 对短轨有软约束保护 (合成 fixture 不会触发删除).
            )  # 计算指标主值和支持字段，支持字段会一起进长表。
            # 按 failure_sample_policy: retain_and_audit 要求，保留失败帧位置信息供审计。
            _long_failure_segments = support_report.get('long_failure_segments')
            if _long_failure_segments:
                failure_segments_by_case_ref[case_ref] = [[int(s), int(e)] for s, e in _long_failure_segments]
            # §9.3 pulse/async 量级门违规聚合 (Round 4 真修复):
            # 每 bundle 抽取 support_report['section9_pulse_async_violations'] 与
            # ['section9_pulse_async_aggregation'], 累加到全维度审计容器, 让下游
            # audit_payload['section9_pulse_async_audit'] 真正可见.
            _section9_violations = support_report.get('section9_pulse_async_violations') or []
            if isinstance(_section9_violations, list):
                for _v in _section9_violations:
                    if not isinstance(_v, Mapping):
                        continue
                    section9_pulse_async_audit['violations_total'] += 1
                    section9_pulse_async_audit['violations_by_method'][bundle_method_name] = (
                        section9_pulse_async_audit['violations_by_method'].get(bundle_method_name, 0) + 1
                    )
                    section9_pulse_async_audit['violations_by_seq'][bundle_seq_id] = (
                        section9_pulse_async_audit['violations_by_seq'].get(bundle_seq_id, 0) + 1
                    )
                    section9_pulse_async_audit['violations_by_case'][case_ref] = (
                        section9_pulse_async_audit['violations_by_case'].get(case_ref, 0) + 1
                    )
                    if bool(_v.get('pulse_violated')):
                        section9_pulse_async_audit['pulse_violated_count'] += 1
                    if bool(_v.get('async_violated')):
                        section9_pulse_async_audit['async_violated_count'] += 1
                    if bool(_v.get('cmp1_cmp5_at_risk')):
                        section9_pulse_async_audit['cmp1_cmp5_at_risk_count'] += 1
                    # 完整违规条目副本 (含原 seq_id/n_pulse/n_async 等), 供审计落盘可追溯
                    section9_pulse_async_audit['violations_detail'].append({
                        'case_ref': case_ref,
                        'seq_id': _v.get('seq_id'),
                        'method_name': bundle_method_name,
                        'n_pulse': _v.get('n_pulse'),
                        'n_pulse_min': _v.get('n_pulse_min'),
                        'n_async': _v.get('n_async'),
                        'n_async_min': _v.get('n_async_min'),
                        'pulse_violated': bool(_v.get('pulse_violated')),
                        'async_violated': bool(_v.get('async_violated')),
                        'cmp1_cmp5_at_risk': bool(_v.get('cmp1_cmp5_at_risk')),
                        'message': str(_v.get('message') or ''),
                    })
            _section9_aggr = support_report.get('section9_pulse_async_aggregation')
            if isinstance(_section9_aggr, Mapping):
                section9_pulse_async_audit['aggregation_by_bundle'].append({
                    'case_ref': case_ref,
                    'seq_id': bundle_seq_id,
                    'method_name': bundle_method_name,
                    'aggregation': dict(_section9_aggr),  # 浅拷贝防 StageResult 内嵌可变引用反向污染
                })
            section9_pulse_async_audit['bundle_count'] += 1
            rows = _build_metric_rows(metric_values)  # 把原始指标字典转成协议要求顺序的 MetricRow 列表。
            for row in rows:  # 每个指标都展开成一行长表记录。
                record = {  # 把当前指标行整理成可写 CSV 的扁平字典。
                    'case_ref': case_ref,  # 这一行属于哪个 case。
                    'seq_id': bundle_seq_id,  # 原始序列编号（已规整为 Python str）。
                    'scene_id': bundle_scene_id,  # 场景编号（已规整为 Python str）。
                    'method_name': bundle_method_name,  # 方法名（已规整为 Python str），后面做分组和统计都要靠它。
                    'metric': row.metric,  # 当前指标名。
                    'value': row.value,  # 当前指标数值。
                    'unit': row.unit,  # 指标单位。
                    'direction': row.direction,  # 指标方向，决定是越大越好还是越小越好。
                    'group': row.group,  # 指标所属组别。
                }
                if bundle_task_id:  # 只有有效 task_id 才写入；D8：使用已规整的 bundle_task_id，避免 is_string_like 检查后再次直接索引 bundle['task_id'] 的双次访问不一致风险，与 repeat_id 同口径。
                    record['task_id'] = bundle_task_id  # task_id 只在有效字符串时写入。
                repeat_id = normalize_optional_string(bundle.get('repeat_id'))  # repeat_id 先规整成干净字符串。
                if repeat_id:  # 有重复编号时也写入长表，方便做配对统计。
                    record['repeat_id'] = repeat_id  # 有重复编号时也写入长表，方便做配对统计。
                record.update({k: v for k, v in support_report.items() if k not in record and k in _SUPPORT_FIELDS})  # 只合并在 _SUPPORT_FIELDS 中声明的支撑字段，过滤掉 ate_degraded/long_failure_segments 等诊断字段。
                metric_rows.append(record)  # 每个 metric row 都追加进统一长表。

        grouped_metric_rows = _group_metric_rows_by_case(metric_rows)  # 按 case_ref 把长表重新分组。
        case_catalog, sweep_rows = _build_case_views(bundles, grouped_metric_rows)  # 生成给统计和选例消费的 case 视图。
        runtime_table_rows = _build_runtime_table(metric_rows)  # 单独抽出运行时指标表，方便画图和审计。

        metrics_path = metrics_dir / 'metric_table.csv'  # 指标 CSV 的最终落盘路径。
        with metrics_path.open('w', encoding='utf-8', newline='') as handle:  # 以文本方式打开 CSV 文件，准备写表头和所有数据行。
            fieldnames = ['case_ref', 'seq_id', 'scene_id', 'method_name']  # CSV 表头先放稳定的身份字段。
            if any('task_id' in row for row in metric_rows):  # 只有存在 task_id 才把这一列写出来。
                fieldnames.append('task_id')  # 只有确实存在 task_id 时才把这一列写进去。
            if any('repeat_id' in row for row in metric_rows):  # 只有存在 repeat_id 才把这一列写出来。
                fieldnames.append('repeat_id')  # repeat_id 同理，只在存在时输出。
            fieldnames.extend(_SUPPORT_FIELDS)  # 支撑字段放在指标字段之前，便于下游读取。
            fieldnames.extend(['metric', 'value', 'unit', 'direction', 'group'])  # 长表核心指标字段放在最后。
            writer = csv.DictWriter(handle, fieldnames=fieldnames)  # 用固定列顺序写 CSV，避免列漂移。
            writer.writeheader()  # 先写表头。
            writer.writerows(metric_rows)  # 再写所有指标长表行。

        priority_metrics = [  # 协议允许用于最终结论的优先指标列表。
            metric_name for metric_name in gate_report.get('conclusion_priority', []) if metric_name in get_metric_order()  # 只保留协议里合法的指标名。
        ]
        # §19.2 I：单一种子定全序不构成本排序成立。
        # 按 §9.3 协议语义：n_seed < n_seed_min 时产生 seed_check 降级标记（violated=True，
        # 标「协议级候选」），供下游（12_run_statistics / 15_build_six_cmp_aggregate）
        # 在宣称全序时拒绝单种子 run。默认软降级而非 raise，保证 eval_pipeline 的
        # 管道功能测试（单种子 bundle 验证数据流）不被误拦；下游主表结论消费方须
        # 校验 seed_check.passed 才能宣称为正式全序。
        # 注意：smoke_without_gt 只是链路验证（build_smoke_only_statistics_payload），
        # 不生产正式主表结论，不生成 seed_check（§19.3「quick 不携带科学语义」）。
        _seed_check = None
        _seed_raise_on_violation = bool((cfg or {}).get('seed_raise_on_violation', False))
        if not smoke_without_gt:
            _protocol_cfg_for_seed = protocol_cfg if hasattr(protocol_cfg, 'get') else None
            _n_seed = _compute_n_seed_from_bundles(bundles)
            _seed_policy = get_seed_policy(_protocol_cfg_for_seed)
            _n_seed_min = int(_seed_policy.get('n_seed_min', 10))
            _seed_violated = _n_seed < _n_seed_min
            if _seed_violated:
                if _seed_raise_on_violation:
                    # 仅当调用方显式要求硬阻（主表结论 run）时 raise。
                    raise ValueError(
                        f"§9.3 N_seed violation: n_seed={_n_seed} < n_seed_min={_n_seed_min}; "
                        f"按 §9.3 「单一种子定全序不构成本排序成立」硬阻止主表统计生成"
                    )
                from liquidloc.common.tee_logger import print_dict as _seed_print  # 局部导入，避免循环。
                _seed_print({
                    'n_seed': _n_seed,
                    'n_seed_min': _n_seed_min,
                    'status': 'protocol_candidate_only',
                    'note': '§9.3 单种子不构成全序结论；seed_check.violated 由下游拒绝作为正式全序',
                }, "§19.2 I seed_check", prefix="[eval]")
            _seed_check = {
                'n_seed': _n_seed,
                'n_seed_min': _n_seed_min,
                'violated': bool(_seed_violated),
                'single_seed_no_conclusion': bool(_seed_policy.get('single_seed_no_conclusion', True)),
                'passed': not _seed_violated,
            }

        # §19.3 M：消融实验过滤与「非主表·非本排序」标注。
        # 判定依据（多信号取真，保证 primary_axis 未从 experiment_cfg 传播时也生效）：
        #   1. eval_pipeline cfg 携带 primary_axis=ablation_variant / experiment_id 含 'ablation'
        #   2. 显式 is_ablation / non_main_table 标志
        #   3. 方法名本身携带消融标记（_wo_ / _ablation_ / ablation_ 前缀）
        # 只要判定为消融实验：消融变体行从主表剔除，且整个输出标注 non_main_table。
        _primary_axis = str((cfg or {}).get('primary_axis') or (protocol_cfg or {}).get('primary_axis') or '')
        _experiment_id = str((cfg or {}).get('experiment_id') or '')
        _is_ablation_experiment = bool(
            str(_primary_axis).lower() in ('ablation_variant', 'ablation')
            or 'ablation' in _experiment_id.lower()
            or bool((cfg or {}).get('is_ablation', False))
            or bool((cfg or {}).get('non_main_table', False))
        )
        _n_ablation_rows_removed = 0  # §19.3：记录本主表剔除的消融行数（供 audit_payload 标注「非主表·非本排序」）。
        if metric_rows:
            _n_before_ablation = len(metric_rows)
            _any_ablation_row = any(_is_ablation_method(row.get('method_name', '')) for row in metric_rows)
            if _is_ablation_experiment or _any_ablation_row:
                metric_rows = [row for row in metric_rows if not _is_ablation_method(row.get('method_name', ''))]
                _n_ablation_rows_removed = _n_before_ablation - len(metric_rows)
            if _n_ablation_rows_removed > 0:
                from liquidloc.common.tee_logger import print_dict as _print_dict  # 局部导入，避免循环。
                _print_dict({
                    'n_rows_before_ablation_filter': _n_before_ablation,
                    'n_rows_after_ablation_filter': len(metric_rows),
                    'removed_ablation_rows': _n_ablation_rows_removed,
                    'is_ablation_experiment': _is_ablation_experiment,
                }, "§19.3 M 消融过滤", prefix="[eval]")

        if smoke_without_gt:  # 缺真值的 smoke 只能验证链路，不能产出正式比较结论。
            statistics_payload = build_smoke_only_statistics_payload()
            best_method = None
            selected_cases = build_empty_selected_cases()
        else:
            statistics_payload = build_statistics_payload(metric_rows, metric_names=priority_metrics)  # 生成统计汇总和显著性检验载荷。
            # §19.2：把 seed_check 写入 statistics_payload，供下游（12_run_statistics / 15_build_six_cmp_aggregate）追溯。
            if _seed_check is not None:
                statistics_payload['seed_check'] = _seed_check
            best_method = select_best_method(  # 根据优先指标和 rmse 选出最优方法。
                statistics_payload['method_summary'],  # 方法名到汇总统计的映射。
                priority_metrics=priority_metrics,  # 本次排序真正使用的优先指标列表。
            )
            case_rules = select_case_rules(case_catalog, priority_metrics=priority_metrics)  # 按指标排序规则生成主样本/失败样本/边界样本规则。
            selected_cases = select_cases(case_catalog, case_rules)  # 交给通用选例器按照规则挑选案例。
        statistics_path = statistics_dir / 'statistics_table.json'  # 统计结果 JSON 的输出路径。
        statistics_path.write_text(dumps_json_text(statistics_payload), encoding='utf-8')  # 直接落盘统计 JSON；dumps_json_text 已在模块顶层导入（D10）。
        cases_path = cases_dir / 'selected_cases.json'  # 选例 JSON 的输出路径。
        cases_path.write_text(dumps_json_text(selected_cases), encoding='utf-8')  # 落盘选例结果。

        plotting_inputs = _build_plotting_inputs(  # 把所有中间结果整理成给绘图脚本直接吃的结构。
            bundles,  # 所有预测 bundle，里面装着每个 case 的主结果。
            gt_records_by_case_ref,  # case_ref -> 真值记录，用于轨迹对比和真值 bundle。
            sweep_rows,  # 已经按 case 拼好的 sweep 行。
            metric_rows,  # 原始长表指标行，用于长表视图和审计。
            runtime_table_rows,  # 运行时统计表，用于性能图和性能审计。
            list(statistics_payload.get('main_table') or []),  # 主表聚合视图供 summary 和主表图公平消费。
        )
        main_table_path = plotting_dir / 'main_table.json'  # 主表聚合视图单独落盘，避免下游误吃 single_run 长表。
        main_table_path.write_text(
            dumps_json_text(plotting_inputs['main_table']),
            encoding='utf-8',
        )  # 主表 JSON 写出完成。
        runtime_table_path = plotting_dir / 'runtime_table.json'  # 绘图输入里的运行时表也单独落一份。
        runtime_table_path.write_text(  # 把运行时表 JSON 单独落一份，方便后续脚本直接读取。
            dumps_json_text(runtime_table_rows),  # JSON 内容保持中文不转义，便于人工查看。
            encoding='utf-8',  # 统一使用 UTF-8 编码。
        )  # 运行时表写出完成。
        sweep_table_path = plotting_dir / 'sweep_table.json'  # sweep 表 JSON 的输出路径。
        sweep_table_path.write_text(  # 把 sweep 表 JSON 单独落一份，供绘图脚本直接消费。
            dumps_json_text(plotting_inputs['sweep_table']),  # 保留中文原样，避免审计时看不懂。
            encoding='utf-8',  # 统一用 UTF-8 编码写盘。
        )  # sweep 表写出完成。
        trajectory_bundle_path = plotting_dir / 'trajectory_bundle.json'  # 轨迹 bundle 的输出路径。
        trajectory_bundle_path.write_text(  # 把轨迹 bundle 单独落盘，给轨迹图和对比图使用。
            dumps_json_text(plotting_inputs['trajectory_bundle']),  # 这里保留完整嵌套结构。
            encoding='utf-8',  # 使用 UTF-8 写盘。
        )  # 轨迹 bundle 写出完成。
        gt_bundle_path = plotting_dir / 'gt_bundle.json'  # 真值 bundle 的输出路径。
        gt_bundle_path.write_text(  # 把真值 bundle 单独落盘，方便轨迹对照和审计。
            dumps_json_text(plotting_inputs['gt_bundle']),  # 真值 bundle 保持原始嵌套结构。
            encoding='utf-8',  # 统一 UTF-8 编码。
        )  # 真值 bundle 写出完成。

        audit_payload = {  # 审计载荷记录本次评估的关键决策和规模。
            'num_prediction_bundles': len(bundles),  # 本次评估输入的 bundle 总数。
            'num_metric_rows': len(metric_rows),  # 最终写出的长表指标行数。
            # §19.3：记录消融行是否被剔除，主表须标明「非主表、非本排序」。
            # 若任一消融行被剔除，本主表已不含消融对照，可追溯审计依据。
            'ablation_rows_removed': _n_ablation_rows_removed,
            # §19.3：整个实验层面标注。仅当 cfg 携带 primary_axis/experiment_id 等消融信号
            # 时置 True；若方法名级已剔除消融变体但实验未显式标注，此处仍为 False 表示
            # 「主表本身不含消融」——两者共同满足 §19.3「明确非主表」的可追溯要求。
            'non_main_table': bool(_is_ablation_experiment),
            'best_method': best_method,  # 最终选出的最优方法。
            'best_method_by_priority': best_method,  # 保留一份同义字段，方便旧下游读取。
            'conclusion_priority': list(priority_metrics),  # D3：拷贝优先指标列表，隔离 audit_payload 与局部 priority_metrics 的引用，防止下游消费者改写 audit_report 时反向污染局部变量；与 _build_plotting_inputs L448 deepcopy 同口径。
            'protocol_gate': deepcopy(gate_report),  # D3：深拷贝协议门控结果，gate_report 含 aggregation_order/conclusion_priority 等嵌套可变列表，浅引用会被下游消费者改写后反向污染；与 _build_plotting_inputs L448 deepcopy 同口径。
            'comparison_eligible': not smoke_without_gt,
            'comparison_status': COMPARISON_STATUS_SMOKE_ONLY_SELF_GROUND_TRUTH if smoke_without_gt else COMPARISON_STATUS_GROUND_TRUTH_BACKED,
            'failure_segments_by_case': deepcopy(failure_segments_by_case_ref),  # D3：深拷贝失败帧位置表，failure_segments_by_case_ref 含嵌套 list[list[int]] 可变结构，浅引用会被下游消费者改写后反向污染；与 _build_plotting_inputs L448 deepcopy 同口径。
            # §9.3 pulse/async 量级门跨 bundle 违规聚合 (2026-07-23 §9 穷举审视 Round 4 真修复):
            # compute_metrics 在 support_report['section9_pulse_async_violations'] /
            # ['section9_pulse_async_aggregation'] 中按单 bundle 生成 §9 违规信息,
            # eval_pipeline 原先只读 _SUPPORT_FIELDS 白名单内的 6 个基础字段, 静默丢弃
            # 这两个 §9 字段 — 导致下游 statistics_table.json / 12_run_statistics 根本看不到
            # 单轨 §9 pulse/async 量级门违规, 无法形成"放松则伤"证据链. 在此聚合后写入
            # audit_payload + 落盘 audits_dir/section9_pulse_async_audit.json.
            'section9_pulse_async_audit': deepcopy(section9_pulse_async_audit),
            # §14 同档/严格优于二阶判定结果：供 12_run_statistics / 人工审计直接读取，
            # 不依赖 audit_payload 嵌套解析；与 section9_pulse_async_audit 同口径落盘。
            'section14_pairwise': deepcopy(statistics_payload.get('section14_pairwise') or []),
        }  # 审计载荷字典结束。
        audit_path = audits_dir / 'eval_audit.json'  # 审计 JSON 的输出路径。
        audit_path.write_text(dumps_json_text(audit_payload), encoding='utf-8')  # 落盘审计报告。
        # §9.3 pulse/async 量级门违规独立落盘 (与 eval_audit.json 同级), 供 12_run_statistics /
        # 人工审计直接读取, 不依赖 audit_payload 嵌套解析.
        section9_audit_path = audits_dir / 'section9_pulse_async_audit.json'
        section9_audit_path.write_text(
            dumps_json_text(section9_pulse_async_audit, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

        artifacts = [  # 统一汇总所有落盘产物路径，返回给上层编排。
            str(metrics_path),  # 指标 CSV。
            str(statistics_path),  # 统计 JSON。
            str(cases_path),  # 选例 JSON。
            str(audit_path),  # 审计 JSON。
            str(section9_audit_path),  # §9.3 pulse/async 量级门违规审计 JSON (Round 4 真修复).
            str(main_table_path),  # 主表聚合 JSON。
            str(runtime_table_path),  # 运行时表 JSON。
            str(sweep_table_path),  # sweep 表 JSON。
            str(trajectory_bundle_path),  # 轨迹 bundle JSON。
            str(gt_bundle_path),  # 真值 bundle JSON。
        ]  # 产物列表结束。

        plotting_metadata = {  # 给绘图脚本和人工审计预留的结构化输入。
            'metric_table': plotting_inputs['metric_table'],  # 给绘图脚本看的指标长表。
            'main_table': plotting_inputs['main_table'],  # 给主表图和 summary 消费的聚合主表。
            'runtime_table': plotting_inputs['runtime_table'],  # 给绘图脚本看的运行时表。
            'sweep_table': plotting_inputs['sweep_table'],  # 给绘图脚本看的 sweep 表。
            'trajectory_bundle': plotting_inputs['trajectory_bundle'],  # 给绘图脚本看的轨迹 bundle。
            'gt_bundle': plotting_inputs['gt_bundle'],  # 给绘图脚本看的真值 bundle。
        }  # 绘图元数据结束。

        return StageResult(  # 把评估阶段的最终结果包装成标准 StageResult。
            stage_name='eval_pipeline',  # 这个阶段名固定标识就是 eval_pipeline。
            artifacts=artifacts,  # 把所有落盘产物路径统一返回给上层。
            metadata={  # metadata 保存中间结构，方便上层继续分析和审计。D3 复核：metric_rows/statistics_payload/selected_cases 虽以直接引用返回，但 plotting_inputs['metric_table'/'runtime_table'/'sweep_table'] 已在 _build_plotting_inputs L447-450 deepcopy 隔离，audit_payload 内部引用已在 L760-764 deepcopy 隔离，StageResult 内不存在跨字段的共享可变引用；与 core_pipeline.py L948-958 同口径（直接引用局部构造对象，不额外 deepcopy）。
                'metric_rows': metric_rows,  # 原始长表指标行，便于上层继续做二次分析。
                'statistics_table': statistics_payload,  # 统计汇总和显著性检验结果。
                'selected_cases': selected_cases,  # 选例结果，给论文展示或人工核验用。
                'audit_report': audit_payload,  # 审计报告，记录本次评估的关键决策信息。
                'plotting_inputs': plotting_metadata,  # 绘图输入，供下游可视化链路直接消费。
                # 偏懒#9 真修复 (Round 3): diagnostics_summary 真聚合到 metadata,
                # 让 candidate_report / scoring_runs / search_audit 聚合层能读到 fallback count.
                'diagnostics_summary': diagnostics_summary,
            },  # metadata 结构结束。
        )  # StageResult 构造完成。



def run(pipeline_cfg: dict | None = None) -> StageResult:  # 保留函数式入口，便于旧脚本直接调用。
    """保留函数式入口，便于旧脚本直接调用。

    Args:  # 参数说明开始。
        pipeline_cfg: 传给 `EvalPipeline.run` 的评估配置。

    Returns:  # 返回值说明开始。
        `EvalPipeline().run(...)` 的原始返回值。
    """
    return EvalPipeline().run(pipeline_cfg)  # 这里不做额外逻辑，纯粹转发到类实现。
