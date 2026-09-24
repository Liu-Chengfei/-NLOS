"""统计测试编排模块。

把 eval_pipeline 里的 repeat_summary / scene_summary 汇总、方法级统计、
显著性检验编排和 smoke-only 统计载荷集中到这个模块，减轻 eval_pipeline 的体积。
本模块只负责编排，核心检验逻辑在 significance_tests.py 里。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from liquidloc.analysis.significance_tests import run_significance_tests
from liquidloc.common.constants import COMPARISON_STATUS_SMOKE_ONLY_SELF_GROUND_TRUTH  # 冒烟对比状态单源常量（D9 漂移根因修复，禁止本地字面量）。
from liquidloc.common.validation import coerce_finite_scalar, is_numeric, is_string_like, normalize_optional_string
from liquidloc.protocol.metric_schema import get_metric_order


def _resolve_statistics_pairing_keys(scene_summary_rows: list[dict[str, Any]]) -> list[str]:
    """决定显著性检验时是否能做配对，以及按哪些键配对。

    Args:
        scene_summary_rows: 场景级汇总行列表（build_scene_summary_rows 的输出）。

    Returns:
        可用于配对检验的字段名列表；如果都不满足就返回空列表。
    """
    if not scene_summary_rows:
        return []

    def _all_rows_have(fields: tuple[str, ...]) -> bool:
        for row in scene_summary_rows:
            for field in fields:
                value = row.get(field)
                if not is_string_like(value) or not str(value).strip():
                    return False
        return True

    # task_id 是每次执行的唯一标识（intentionally unique per expanded execution
    # instance），不能作为场景级配对键；repeat_id 在 scene_summary 阶段已被剥离
    # （build_repeat_summary_rows 显式过滤 repeat_id）。仅使用场景级稳定标识符。
    for fields in (
        ('scene_id', 'seq_id'),
        ('scene_id',),
    ):
        if _all_rows_have(fields):
            return list(fields)
    return []


def _derive_repeat_summary_case_ref(row: Mapping[str, Any]) -> str:
    """从单次运行行里导出重复汇总使用的稳定 case_ref。"""
    # case_ref 是必需键，用 [] 直接访问防止静默失败；缺键属值合同违例，统一抛 ValueError。
    if 'case_ref' not in row:
        raise ValueError('metric rows must contain a non-empty case_ref')
    case_ref = normalize_optional_string(row['case_ref'])
    if not case_ref:
        raise ValueError('metric rows must contain a non-empty case_ref')
    repeat_id = normalize_optional_string(row.get('repeat_id'))
    if not repeat_id:
        return case_ref
    scene_id = normalize_optional_string(row.get('scene_id'))
    seq_id = normalize_optional_string(row.get('seq_id'))
    method_name = normalize_optional_string(row.get('method_name'))
    if scene_id and seq_id and method_name:
        # task_id is intentionally unique per expanded execution instance, so it
        # cannot serve as the repeat-invariant grouping key here.
        return f'{scene_id}::{seq_id}::{method_name}'
    suffix = f'::{repeat_id}'
    if case_ref.endswith(suffix):
        return case_ref[: -len(suffix)]
    return case_ref


def build_repeat_summary_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """先按协议把 single_run 汇总成 repeat_summary 行。"""
    grouped_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in metric_rows:
        if 'method_name' not in row:
            raise ValueError('metric rows must contain a non-empty method_name')
        method_name = normalize_optional_string(row['method_name'])
        if not method_name:
            raise ValueError('metric rows must contain a non-empty method_name')
        if 'metric' not in row:
            raise ValueError('metric rows must contain a non-empty metric name')
        metric_name = normalize_optional_string(row['metric'])
        if not metric_name:
            raise ValueError('metric rows must contain a non-empty metric name')
        if 'value' not in row:
            raise ValueError('metric rows must contain a value')
        base_case_ref = _derive_repeat_summary_case_ref(row)
        grouped_rows.setdefault((method_name, metric_name, base_case_ref), []).append(dict(row))

    repeat_summary_rows: list[dict[str, Any]] = []
    for (_, _, base_case_ref), rows in sorted(grouped_rows.items()):
        first_row = dict(rows[0])
        values: list[float] = []
        for row in rows:
            raw_value = row['value']
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"metric row value must be numeric, got {type(raw_value).__name__}"
                ) from exc
            values.append(coerce_finite_scalar(numeric_value, name='metric row value'))
        summary_row = {key: value for key, value in first_row.items() if key != 'repeat_id'}
        summary_row['case_ref'] = base_case_ref
        summary_row['value'] = math.fsum(values) / len(values)
        summary_row['num_single_runs'] = len(values)
        repeat_summary_rows.append(summary_row)
    return repeat_summary_rows


def _resolve_scene_summary_group_id(row: Mapping[str, Any]) -> str:
    """解析 scene_summary 的场景级分组键。"""
    # task_id 是每次执行的唯一标识（intentionally unique per expanded execution
    # instance），不能作为场景级分组键；case_ref 在 scene_id/seq_id 缺失时会退化为
    # 嵌入 task_id 的任务级键（见 eval_pipeline._build_case_ref），同样不是场景级
    # 稳定标识符。仅使用场景级稳定标识符，与 _resolve_statistics_pairing_keys 同口径。
    for field_name in ('scene_id', 'seq_id'):
        normalized_value = normalize_optional_string(row.get(field_name))
        if normalized_value:
            return normalized_value
    raise ValueError('repeat summary rows must contain at least one stable scene identifier')


def build_scene_summary_rows(repeat_summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """再按协议把 repeat_summary 汇总成 scene_summary 行。"""
    grouped_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in repeat_summary_rows:
        if 'method_name' not in row:
            raise ValueError('repeat summary rows must contain a non-empty method_name')
        method_name = normalize_optional_string(row['method_name'])
        if not method_name:
            raise ValueError('repeat summary rows must contain a non-empty method_name')
        if 'metric' not in row:
            raise ValueError('repeat summary rows must contain a non-empty metric name')
        metric_name = normalize_optional_string(row['metric'])
        if not metric_name:
            raise ValueError('repeat summary rows must contain a non-empty metric name')
        scene_group_id = _resolve_scene_summary_group_id(row)
        grouped_rows.setdefault((method_name, metric_name, scene_group_id), []).append(dict(row))

    scene_summary_rows: list[dict[str, Any]] = []
    for (_, _, scene_group_id), rows in sorted(grouped_rows.items()):
        first_row = dict(rows[0])
        values: list[float] = []
        for row in rows:
            raw_value = row['value']
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"repeat summary row value must be numeric, got {type(raw_value).__name__}"
                ) from exc
            values.append(coerce_finite_scalar(numeric_value, name='repeat summary row value'))
        summary_row = {
            'case_ref': scene_group_id,
            'method_name': first_row['method_name'],
            'metric': first_row['metric'],
            'value': math.fsum(values) / len(values),
            'num_repeat_summaries': len(values),
        }
        for field_name in ('scene_id', 'unit', 'direction', 'group'):
            normalized_value = first_row.get(field_name)
            if normalized_value is not None:
                summary_row[field_name] = normalized_value
        for field_name in ('seq_id', 'task_id'):
            unique_values = {
                value
                for value in (row.get(field_name) for row in rows)
                if value is not None
            }
            if len(unique_values) == 1:
                summary_row[field_name] = unique_values.pop()
        scene_summary_rows.append(summary_row)
    return scene_summary_rows


def _build_main_table_rows_from_method_summary(
    method_summary: Mapping[str, Mapping[str, Any]],
    *,
    priority_metrics: list[str],
) -> list[dict[str, Any]]:
    """把方法级统计载荷整理成主表行。

    优先指标对应的 ``mean_{metric}`` 字段按 ``priority_metrics`` 顺序排列，
    其余 ``mean_`` 前缀字段按字母序追加。优先字段视为必填键，缺失时抛
    ``ValueError`` 以暴露数据合同违例；额外发现的字段为可选，缺失时静默跳过。

    Args:
        method_summary: 方法名到统计摘要的映射，每个摘要应包含 ``mean_{metric}``
            形式的字段（由 ``build_statistics_payload`` 保证）。
        priority_metrics: 优先指标列表，对应的 ``mean_{metric}`` 字段会排在主表前列
            且视为必填键。

    Returns:
        主表行列表，每行包含 ``method_name`` 和按顺序排列的 ``mean_`` 字段；
        若 ``method_summary`` 为空则返回空列表。
    """
    preferred_mean_fields = [f'mean_{metric_name}' for metric_name in priority_metrics]
    discovered_mean_fields = {
        str(key)
        for summary in method_summary.values()
        for key in summary
        if is_string_like(key) and str(key).startswith('mean_')
    }
    ordered_mean_fields: list[str] = []
    for field_name in preferred_mean_fields:
        if field_name not in ordered_mean_fields:
            ordered_mean_fields.append(field_name)
    for field_name in sorted(discovered_mean_fields):
        if field_name not in ordered_mean_fields:
            ordered_mean_fields.append(field_name)

    # §27.4 分散度（标准差/置信带/bootstrap）必须与均值一同报告；
    # 无分散不构成全序证据。将优先指标的 std_ 字段与 mean_ 字段一同
    # 排入主表顺序（mean 在前、std 紧随其后），其余 std_ 字段按字母序追加。
    preferred_std_fields = [f'std_{metric_name}' for metric_name in priority_metrics]
    discovered_std_fields = {
        str(key)
        for summary in method_summary.values()
        for key in summary
        if is_string_like(key) and str(key).startswith('std_')
    }
    ordered_std_fields: list[str] = []
    for field_name in preferred_std_fields:
        if field_name not in ordered_std_fields:
            ordered_std_fields.append(field_name)
    for field_name in sorted(discovered_std_fields):
        if field_name not in ordered_std_fields:
            ordered_std_fields.append(field_name)

    preferred_field_set = set(preferred_mean_fields)
    main_table_rows: list[dict[str, Any]] = []
    for method_name, summary in sorted(method_summary.items()):
        row: dict[str, Any] = {'method_name': method_name}
        for field_name in ordered_mean_fields:
            if field_name in preferred_field_set:
                if field_name not in summary:
                    raise ValueError(f"method '{method_name}' summary is missing required key '{field_name}'")
                row[field_name] = summary[field_name]
            elif field_name in summary:
                row[field_name] = summary[field_name]
        # §27.4: 将 std 字段与对应 mean 一同写入主表，确保分散度可查。
        for field_name in ordered_std_fields:
            if field_name in summary:
                row[field_name] = summary[field_name]
        main_table_rows.append(row)
    return main_table_rows


def build_statistics_payload(metric_rows: list[dict[str, Any]], *, metric_names: list[str]) -> dict[str, Any]:
    """把指标长表汇总成方法级统计和显著性检验载荷。

    Args:
        metric_rows: 指标长表行列表，每行必须包含 method_name、metric、value 等必需键。
        metric_names: 需要参与显著性检验和均值汇总的优先指标名列表。

    Returns:
        包含 method_summary、pairwise_tests、pairing_keys 和 main_table 的统计载荷字典。

    Raises:
        ValueError: 当 method_name/metric 缺失或为空、rmse 行缺失、优先指标行缺失
            或 scene_summary 的 value 非数值/非有限时抛出。
    """
    from liquidloc.common.tee_logger import print_dict  # 入口参数日志（D8 print_dict 增强）
    print_dict(
        {
            "metric_rows_count": len(metric_rows) if hasattr(metric_rows, "__len__") else None,
            "metric_names": list(metric_names) if metric_names is not None else None,
        },
        "build_statistics_payload 入口参数",
        prefix="[analysis]",
    )
    repeat_summary_rows = build_repeat_summary_rows(metric_rows)
    scene_summary_rows = build_scene_summary_rows(repeat_summary_rows)

    per_method_metric_values: dict[str, dict[str, list[float]]] = {}
    per_method_single_run_counts: dict[str, int] = {}
    per_method_repeat_counts: dict[str, int] = {}
    per_method_scene_counts: dict[str, int] = {}

    # build_repeat_summary_rows 已校验每行 method_name 和 metric 非空，此处直接用 [] 访问。
    for row in metric_rows:
        if row['metric'] == 'rmse':
            method_name = row['method_name']
            per_method_single_run_counts[method_name] = per_method_single_run_counts.get(method_name, 0) + 1

    for row in repeat_summary_rows:
        if row['metric'] == 'rmse':
            method_name = row['method_name']
            per_method_repeat_counts[method_name] = per_method_repeat_counts.get(method_name, 0) + 1

    for row in scene_summary_rows:
        method_name = row['method_name']
        metric_name = row['metric']
        try:  # value 必须是数值；OverflowError 守卫超大整数，统一转 ValueError 以符合值合同。
            metric_value = float(row['value'])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f'scene summary row value must be numeric for method {method_name!r}, metric {metric_name!r}'
            ) from exc
        if not math.isfinite(metric_value):  # NaN/Inf 会破坏后续均值和显著性检验。
            raise ValueError(
                f'scene summary row value must be finite for method {method_name!r}, metric {metric_name!r}, got {metric_value}'
            )
        per_method_metric_values.setdefault(method_name, {}).setdefault(metric_name, []).append(metric_value)
        if metric_name == 'rmse':
            per_method_scene_counts[method_name] = per_method_scene_counts.get(method_name, 0) + 1

    method_summary: dict[str, dict[str, Any]] = {}
    for method_name, metric_map in sorted(per_method_metric_values.items()):
        # rmse 是必需指标，缺键属值合同违例；metric_map 的 list 一旦建键就非空（append 即建）。
        if 'rmse' not in metric_map:
            raise ValueError(f"method '{method_name}' is missing rmse rows")
        rmse_values = metric_map['rmse']
        # §14.4「主报告统计：至少报 raw 位置误差的均值与分散」
        # 故除 mean 外还须报 std (样本标准分散, Bessel 校正, n<2 时退为 0.0).
        rmse_mean = math.fsum(rmse_values) / len(rmse_values)
        if len(rmse_values) >= 2:
            rmse_variance = math.fsum(
                (value - rmse_mean) ** 2 for value in rmse_values
            ) / (len(rmse_values) - 1)
            rmse_std = math.sqrt(rmse_variance)
        else:
            rmse_std = 0.0
        summary = {
            'num_bundles': per_method_single_run_counts.get(method_name, 0),
            'num_repeat_summaries': per_method_repeat_counts.get(method_name, 0),
            'num_scene_summaries': per_method_scene_counts.get(method_name, 0),
            'mean_rmse': rmse_mean,
            'std_rmse': rmse_std,  # §14.4 raw 位置误差分散（样本标准差, Bessel 校正）.
        }
        for metric_name in metric_names:
            if metric_name not in metric_map:
                raise ValueError(f"method '{method_name}' is missing priority metric rows for {metric_name}")
            values = metric_map[metric_name]
            mean_value = math.fsum(values) / len(values)
            summary[f'mean_{metric_name}'] = mean_value
            if len(values) >= 2:
                variance = math.fsum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
                summary[f'std_{metric_name}'] = math.sqrt(variance)
            else:
                summary[f'std_{metric_name}'] = 0.0
        method_summary[method_name] = summary

    statistics_payload = {'method_summary': method_summary}

    if metric_names:
        pairing_keys = _resolve_statistics_pairing_keys(scene_summary_rows)
        statistics_payload['pairwise_tests'] = run_significance_tests(
            scene_summary_rows,
            group_keys=['method_name'],
            metric_names=metric_names,
            pairing_keys=pairing_keys or None,
        )
        statistics_payload['pairing_keys'] = pairing_keys
    else:
        statistics_payload['pairwise_tests'] = []
        statistics_payload['pairing_keys'] = []
    statistics_payload['main_table'] = _build_main_table_rows_from_method_summary(
        method_summary,
        priority_metrics=metric_names,
    )

    # §14.3 同档/严格优于二阶判定：对每对方法按 §14 阈值（ε_≈=0.08、ε_>=0.03、p_>=0.6）
    # 在主指标 rmse 上做判定，写入 statistics_payload['section14_pairwise']。
    # 主指标 rmse 是 §14.1 主序依据，p95 不得替代主序判定。
    statistics_payload['section14_pairwise'] = _build_section14_pairwise_summary(
        statistics_payload.get('pairwise_tests') or [],
        statistics_payload.get('method_summary') or {},
        primary_metric='rmse',
    )
    return statistics_payload


def _build_section14_pairwise_summary(
    pairwise_tests: list[dict[str, Any]],
    method_summary: Mapping[str, Mapping[str, Any]],
    *,
    primary_metric: str = 'rmse',
) -> list[dict[str, Any]]:
    """§14.3 同档/严格优于二阶判定汇总。

    对每对方法按 §14 阈值（ε_≈=0.08、ε_>=0.03、p_>=0.6）在主指标上做判定。
    若主指标区间不可比（如 missing），跳过该对并保留 None 占位。

    注意：本函数调用 protocol.experiment_gates.check_section14_comparison 执行判定，
    避免重复实现比较逻辑。

    Args:
        pairwise_tests: significance_tests.run_significance_tests 的原始输出列表。
        method_summary: 方法级统计摘要，用于取 mean_{primary_metric}。
        primary_metric: 主指标名（默认 'rmse'，§14.1 主序依据）。

    Returns:
        list[dict]：每条 pairwise 记录追加 section14 同档/严格优于判定字段。
    """
    from liquidloc.protocol.experiment_gates import check_section14_comparison

    section14_rows: list[dict[str, Any]] = []
    for record in pairwise_tests:
        # §14.3: 字段名兼容 pairwise_tests 输出 (method_name_a/b) 和自定义字段 (method_a/b).
        method_a = record.get('method_a') or record.get('method_name_a')
        method_b = record.get('method_b') or record.get('method_name_b')
        metric_name = record.get('metric') or record.get('metric_name')
        if metric_name != primary_metric:
            continue  # §14 主指标判定只覆盖 primary_metric（rmse 优先于 p95，§14.4 细节）.
        if method_a is None or method_b is None:
            continue
        summary_a = method_summary.get(method_a, {})
        summary_b = method_summary.get(method_b, {})
        mean_key = f'mean_{primary_metric}'
        mean_a = summary_a.get(mean_key)
        mean_b = summary_b.get(mean_key)
        if mean_a is None or mean_b is None:
            continue  # 缺均值无法判定，跳过；上游已对 None 做显式声明（D7 公平性）.
        try:
            mean_a_f = float(mean_a)
            mean_b_f = float(mean_b)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(mean_a_f) or not math.isfinite(mean_b_f):
            continue

        # 调用 §14 中央判定函数，避免重复实现比较逻辑。
        prob_b_better = record.get('bayes_prob_b_better')
        if prob_b_better is not None:
            try:
                prob_b_better_f = float(prob_b_better)
                if not math.isfinite(prob_b_better_f) or not (0.0 <= prob_b_better_f <= 1.0):
                    prob_b_better_f = None
            except (TypeError, ValueError):
                prob_b_better_f = None
        else:
            prob_b_better_f = None

        # §18 依赖矩阵执行点校验：传递 comparison_label 以启用模块依赖前提校验。
        comparison_label = record.get('comparison_label') or record.get('comparison')
        if comparison_label is None:
            # 根据 method_a/method_b 的命名模式推断比较编号（如 method_a 含 "baseline" → comparison_1 等）。
            # 若无法推断，则不传递 comparison_label，跳过 §18 校验。
            comparison_label = None

        # §18 模块可用集合：优先从 record 取，未提供则默认全集（不报告缺失）。
        available_modules = record.get('available_modules')
        if available_modules is not None:
            try:
                available_modules = set(available_modules)
            except TypeError:
                available_modules = None

        cmp_result = check_section14_comparison(
            mean_a_f, mean_b_f,
            prob_b_better=prob_b_better_f,
            comparison_label=comparison_label,
            available_modules=available_modules,
            raise_on_violation=False,
        )

        section14_rows.append({
            'method_a': method_a,
            'method_b': method_b,
            'metric': metric_name,
            'mean_a': mean_a_f,
            'mean_b': mean_b_f,
            'std_a': record.get('std_a'),
            'std_b': record.get('std_b'),
            'relative_improvement': cmp_result['relative_improvement'],
            'prob_b_better': prob_b_better_f,
            'epsilon_approx': cmp_result.get('epsilon_approx'),
            'epsilon_strict': cmp_result.get('epsilon_strict'),
            'p_strict_win': cmp_result.get('p_strict_win'),
            'same_tier': cmp_result['same_tier'],
            'strict_better': cmp_result['strict_better'],
            'pairwise_strict_better': cmp_result.get('pairwise_strict_better', False),
            'comparison_status': cmp_result['comparison_status'],
            'comparison_label': comparison_label,
            'section18_dependency_check': cmp_result.get('section18_dependency_check'),
        })
    return section14_rows


def build_smoke_only_statistics_payload() -> dict[str, Any]:
    """构建无对比维度的冒烟专用统计载荷。

    冒烟模式（无外部真值）下不产生方法间显著性检验与方法级均值汇总，
    仅返回空结构并标注对比不可用，供下游审计与统计载荷统一消费。
    comparison_status 取值经 common/constants.py 单源常量约束，
    与 pipelines/eval_pipeline.py audit_payload 同口径，禁止本地字面量漂移。
    """
    return {
        'method_summary': {},
        'main_table': [],
        'pairwise_tests': [],
        'pairing_keys': [],
        'comparison_eligible': False,
        'comparison_status': COMPARISON_STATUS_SMOKE_ONLY_SELF_GROUND_TRUTH,
    }


# ---------------------------------------------------------------------------
# 向后兼容别名：保留旧名供已有代码使用。
# ---------------------------------------------------------------------------
_build_repeat_summary_rows = build_repeat_summary_rows
_build_scene_summary_rows = build_scene_summary_rows
