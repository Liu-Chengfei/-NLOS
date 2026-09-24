"""验证高层消费者能否接受冻结后的评估产物。

这个脚本负责把核心评估流水线产出的预测索引、指标表、统计表、选例和
绘图输入同步到验证目录，然后逐个调用 summary 构建器和各个绘图函数，
确认这些高层消费者仍然能正确处理冻结输入并产出有效产物。

上游依赖：08_run_core_experiments.py 或 09_run_extended_experiments.py 产出的
评估产物目录（包含 prediction_index.json、metric_table.csv 等）。
下游调用者：15_audit_outputs.py 会检查本脚本的输出是否完整。
核心变量：OUTPUT_ROOT 验证输出根目录，所有同步和新生成的产物都落在这里。
"""

from __future__ import annotations  # 允许使用更现代的类型注解写法。

import argparse  # 解析命令行参数。
import csv  # 读取 CSV 格式的指标表。
import json  # 读写 JSON 文件。
import shutil  # 复制文件用于同步评估产物。
import sys  # 调整导入路径。
from collections.abc import Mapping  # 判断映射类型。
from pathlib import Path  # 统一处理文件路径。
from typing import Any  # 标注任意 JSON 风格结构。

ROOT = Path(__file__).resolve().parents[1]  # 从脚本位置反推仓库根目录。
SRC = ROOT / "src"  # 保存源码目录。
LOCAL_PY_DEPS = ROOT / "outputs" / ".python-deps"  # 本地 Python 依赖目录，某些环境下需要从这里导入包。
for candidate in (LOCAL_PY_DEPS, SRC):  # 依次检查两个候选目录。
    if candidate.is_dir():  # 只有目录存在才加入导入路径。
        candidate_str = str(candidate)  # 转成字符串，方便后续比较。
        if candidate_str not in sys.path:  # 避免重复添加。
            sys.path.insert(0, candidate_str)  # 把候选目录插到导入路径最前面。

from liquidloc.analysis.summary_builder import build_summary  # 构建 summary 的函数。
from liquidloc.plotting.plot_calibration import render_calibration_figure  # 渲染校准图。
from liquidloc.plotting.plot_cases import render_case_figures  # 渲染案例图。
from liquidloc.plotting.plot_main_table import render_main_table_figure  # 渲染主表图。
from liquidloc.plotting.plot_runtime import render_runtime_figure  # 渲染运行时图。
from liquidloc.plotting.plot_sweeps import render_sweep_figure  # 渲染扫描图。
from liquidloc.plotting.plot_trajectories import render_trajectory_figure  # 渲染轨迹图。

OUTPUT_ROOT = ROOT / "outputs" / "high_level_consumer_verify"  # 默认验证输出根目录。
INPUT_ROOT = ROOT / "outputs" / "core_script_smoke"  # 默认输入根目录，指向核心脚本冒烟输出。
INPUT_INDEX_PATH = INPUT_ROOT / "audits" / "prediction_index.json"  # 默认预测索引路径。
GROUND_TRUTH_ROOT = ROOT / "tests" / "fixtures" / "datasets" / "miluv"  # 默认真值根目录，使用 MILUV fixture。
EVAL_STAGE_NAME = "eval_pipeline"  # 评估阶段名称，用于最终报告。


def _print_summary(payload: dict[str, Any]) -> int:
    """打印 JSON 摘要并返回退出码。"""
    from liquidloc.common.io_utils import dumps_json_text
    print(dumps_json_text(payload))  # 打印 JSON 格式的摘要。
    return int(payload["exit_code"])  # 从摘要中提取退出码返回。


def _fail(stage: str, blocker: str, detail: str) -> int:
    """构造失败摘要并打印返回。"""
    return _print_summary(  # 打印失败摘要并返回退出码。
        {  # 构造失败摘要对象。
            "status": "failed",  # 状态标记为失败。
            "stage": stage,  # 失败发生的阶段名。
            "blocker": blocker,  # 失败的阻塞原因。
            "detail": detail,  # 失败的详细描述。
            "exit_code": 1,  # 失败退出码。
        }
    )  # 失败摘要构造结束。


def _load_json(path: Path) -> Any:
    """读取 JSON 文件并返回反序列化结果。"""
    from liquidloc.common.io_utils import read_json
    return read_json(path)  # 按 UTF-8 读取并解析 JSON。


def _resolve_prediction_path(prediction_path: Path, *, index_path: Path) -> Path:
    """把预测路径解析成绝对路径，相对路径按索引文件所在目录解释。"""
    if prediction_path.is_absolute():  # 已经是绝对路径就直接返回。
        return prediction_path  # 返回原始绝对路径。
    return (index_path.parent / prediction_path).resolve()  # 相对路径按索引文件目录补全。


def _validate_prediction_bundle(
    payload: Any,
    *,
    index_path: Path,
    ground_truth_root: Path,
    entry_index: int,
) -> dict[str, Any]:
    """验证单个预测 bundle 的结构和关联文件是否完整。

    检查预测 bundle 必须包含 seq_id、scene_id、method_name、prediction_path，
    并且 prediction_path 指向的文件和对应的真值文件都必须存在。
    """
    if not isinstance(payload, dict):  # bundle 必须是字典。
        raise TypeError(f"prediction bundle {entry_index} in {index_path} must be a mapping")  # 类型不对就报错。
    required_keys = ("seq_id", "scene_id", "method_name", "prediction_path")  # 必须包含的键。
    missing_keys = [key for key in required_keys if key not in payload]  # 找出缺失的键。
    if missing_keys:  # 有缺失键就报错。
        raise KeyError(f"prediction bundle {entry_index} missing required keys: {missing_keys}")  # 明确指出缺了什么。
    seq_id = payload.get("seq_id")  # 取出序列 ID。
    if not isinstance(seq_id, str) or not seq_id.strip():  # seq_id 必须是非空字符串。
        raise TypeError(f"prediction bundle {entry_index} seq_id must be a non-empty string")  # 格式不对就报错。
    prediction_path = payload.get("prediction_path")  # 取出预测文件路径。
    if not isinstance(prediction_path, (str, Path)):  # prediction_path 必须是路径类型。
        raise TypeError(f"prediction bundle {entry_index} prediction_path must be path-like")  # 类型不对就报错。
    prediction_path = _resolve_prediction_path(Path(prediction_path), index_path=index_path)  # 解析成绝对路径。
    if not prediction_path.is_file():  # 预测文件必须存在。
        raise FileNotFoundError(f"prediction bundle {entry_index} prediction_path missing: {prediction_path}")  # 文件不存在就报错。
    prediction_payload = _load_json(prediction_path)  # 读取预测文件内容。
    if not isinstance(prediction_payload, dict):  # 预测内容必须是映射。
        raise TypeError(f"prediction bundle {entry_index} must contain a mapping payload")  # 不是映射就报错。
    required_payload_keys = ("seq_id", "scene_id", "method_name", "states")  # 预测内容必须包含的键。
    missing_payload_keys = [key for key in required_payload_keys if key not in prediction_payload]  # 找出缺失键。
    if missing_payload_keys:  # 有缺失键就报错。
        raise KeyError(f"prediction bundle {entry_index} payload missing required keys: {missing_payload_keys}")  # 明确指出缺了什么。
    states = prediction_payload.get("states")  # 取出状态序列。
    if not isinstance(states, list) or not states:  # 状态序列必须是非空列表。
        raise ValueError(f"prediction bundle {entry_index} payload must contain a non-empty states list")  # 不符合就报错。
    gt_path = ground_truth_root / seq_id / "gt.json"  # 构造真值文件路径。
    if not gt_path.is_file():  # 真值文件必须存在。
        raise FileNotFoundError(f"ground truth missing for seq_id {seq_id}: {gt_path}")  # 不存在就报错。
    gt_payload = _load_json(gt_path)  # 读取真值文件内容。
    if not isinstance(gt_payload, list) or not gt_payload:  # 真值内容必须是非空列表。
        raise ValueError(f"ground truth payload for seq_id {seq_id} must be a non-empty list")  # 不符合就报错。
    return {  # 返回验证通过的摘要信息。
        "seq_id": seq_id,  # 序列 ID。
        "scene_id": prediction_payload["scene_id"],  # 场景 ID。
        "method_name": prediction_payload["method_name"],  # 方法名。
        "prediction_path": prediction_path,  # 预测文件路径。
        "gt_path": gt_path,  # 真值文件路径。
    }  # 验证摘要结束。


def _validate_fixed_route_inputs(
    *,
    index_path: Path,
    ground_truth_root: Path,
) -> list[dict[str, Any]]:
    """验证固定路线的所有输入是否完整。

    读取预测索引文件，逐条验证每个预测 bundle 的结构和关联文件。
    """
    if not index_path.is_file():  # 索引文件必须存在。
        raise FileNotFoundError(f"prediction_index missing: {index_path}")  # 不存在就报错。
    index_payload = _load_json(index_path)  # 读取索引文件内容。
    if not isinstance(index_payload, list) or not index_payload:  # 索引必须是非空列表。
        raise ValueError("prediction_index must be a non-empty list")  # 不符合就报错。
    return [  # 逐条验证并收集结果。
        _validate_prediction_bundle(  # 验证单个预测 bundle。
            entry,  # 当前条目。
            index_path=index_path,  # 索引文件路径，用于解析相对路径。
            ground_truth_root=ground_truth_root,  # 真值根目录。
            entry_index=entry_index,  # 条目序号，用于报错定位。
        )
        for entry_index, entry in enumerate(index_payload)  # 遍历所有索引条目。
    ]  # 验证结果列表结束。


def _sync_file(source_path: Path, destination_path: Path) -> None:
    """把源文件复制到目标路径，自动创建目标目录。"""
    destination_path.parent.mkdir(parents=True, exist_ok=True)  # 先确保目标目录存在。
    shutil.copy2(source_path, destination_path)  # 复制文件并保留元数据。


def _sync_fixed_eval_route(source_eval_root: Path, output_root: Path) -> dict[str, str]:
    """把固定评估路线的关键产物同步到验证目录。

    把评估目录下的审计报告、指标表、统计表、选例和绘图输入
    全部复制到 output_root 对应的子目录中。
    """
    route_files = {  # 定义需要同步的文件映射，键为相对路径，值为目标路径。
        "audits/eval_audit.json": output_root / "audits" / "eval_audit.json",  # 评估审计报告。
        "metrics/metric_table.csv": output_root / "metrics" / "metric_table.csv",  # 指标表 CSV。
        "statistics/statistics_table.json": output_root / "statistics" / "statistics_table.json",  # 统计表 JSON。
        "cases/selected_cases.json": output_root / "cases" / "selected_cases.json",  # 选例文件。
        "plotting_inputs/main_table.json": output_root / "plotting_inputs" / "main_table.json",  # 主表聚合视图。
        "plotting_inputs/runtime_table.json": output_root / "plotting_inputs" / "runtime_table.json",  # 运行时表。
        "plotting_inputs/sweep_table.json": output_root / "plotting_inputs" / "sweep_table.json",  # 扫描表。
        "plotting_inputs/trajectory_bundle.json": output_root / "plotting_inputs" / "trajectory_bundle.json",  # 轨迹数据。
        "plotting_inputs/gt_bundle.json": output_root / "plotting_inputs" / "gt_bundle.json",  # 真值数据。
    }  # 文件映射结束。
    synced_files: dict[str, str] = {}  # 记录已同步的文件路径。
    for relative_path, destination_path in route_files.items():  # 逐个文件同步。
        source_path = source_eval_root / relative_path  # 拼出源文件路径。
        if not source_path.is_file():  # 源文件必须存在。
            raise FileNotFoundError(f"fixed eval artifact missing: {source_path}")  # 不存在就报错。
        _sync_file(source_path, destination_path)  # 执行文件复制。
        synced_files[relative_path] = str(destination_path)  # 记录已同步路径。
    return synced_files  # 返回同步结果。


def _load_metric_rows(metric_path: Path) -> list[dict[str, Any]]:
    """读取 CSV 格式的指标表并验证必须列是否齐全。"""
    with metric_path.open("r", encoding="utf-8", newline="") as handle:  # 打开 CSV 文件。
        metric_rows = list(csv.DictReader(handle))  # 把 CSV 读成字典列表。
    if not metric_rows:  # 指标表不能为空。
        raise ValueError("metric_table.csv must be non-empty")  # 空表就报错。
    required_keys = {"case_ref", "seq_id", "scene_id", "method_name", "metric", "value"}  # 必须包含的列名。
    missing_keys = sorted(required_keys - set(metric_rows[0].keys()))  # 找出缺失的列。
    if missing_keys:  # 有缺失列就报错。
        raise KeyError(f"metric_table.csv missing required columns: {missing_keys}")  # 明确指出缺了什么。
    return metric_rows  # 返回指标行列表。


def _load_statistics_payload(statistics_path: Path) -> dict[str, Any]:
    """读取统计表 JSON 并验证必须字段是否齐全。"""
    statistics_payload = _load_json(statistics_path)  # 读取统计表内容。
    if not isinstance(statistics_payload, dict):  # 统计表必须是映射。
        raise TypeError("statistics_table.json must contain a mapping")  # 不是映射就报错。
    required_keys = ("method_summary", "pairwise_tests")  # 必须包含的键。
    missing_keys = [key for key in required_keys if key not in statistics_payload]  # 找出缺失键。
    if missing_keys:  # 有缺失键就报错。
        raise KeyError(f"statistics_table.json missing required keys: {missing_keys}")  # 明确指出缺了什么。
    return statistics_payload  # 返回统计表内容。


def _load_selected_cases(cases_path: Path) -> dict[str, Any]:
    """读取选例文件并验证必须分组是否齐全。"""
    selected_cases = _load_json(cases_path)  # 读取选例内容。
    if not isinstance(selected_cases, dict):  # 选例必须是映射。
        raise TypeError("selected_cases.json must contain a mapping")  # 不是映射就报错。
    required_groups = ("main_cases", "failure_cases", "boundary_cases")  # 必须包含的分组。
    missing_groups = [name for name in required_groups if name not in selected_cases]  # 找出缺失分组。
    if missing_groups:  # 有缺失分组就报错。
        raise KeyError(f"selected_cases.json missing required groups: {missing_groups}")  # 明确指出缺了什么。
    return selected_cases  # 返回选例内容。


def _load_runtime_table(runtime_path: Path) -> list[dict[str, Any]]:
    """读取运行时表 JSON 并验证必须列是否齐全。"""
    runtime_rows = _load_json(runtime_path)  # 读取运行时表内容。
    if not isinstance(runtime_rows, list) or not runtime_rows:  # 运行时表必须是非空列表。
        raise ValueError("runtime_table.json must contain a non-empty list")  # 不符合就报错。
    required_keys = {"case_ref", "latency_mean", "latency_p50", "latency_p95", "params", "ram_peak"}  # 必须包含的列名。
    missing_keys = sorted(required_keys - set(runtime_rows[0].keys()))  # 找出缺失的列。
    if missing_keys:  # 有缺失列就报错。
        raise KeyError(f"runtime_table.json missing required columns: {missing_keys}")  # 明确指出缺了什么。
    return runtime_rows  # 返回运行时行列表。


def _resolve_plotting_root(output_root: Path) -> Path:
    """返回绘图消费检查的输出目录。"""
    return output_root / "plotting_consume_check"  # 绘图产物落在这里。


def _resolve_source_eval_root(input_root: Path, input_index_path: Path) -> Path:
    """定位评估产物的源目录。

    优先在 input_root/eval 下找，找不到就往上层目录找 eval 子目录。
    """
    candidate = input_root / "eval"  # 先看输入根目录下有没有 eval 子目录。
    if candidate.is_dir():  # 找到就直接用。
        return candidate  # 返回 input_root/eval。
    sibling_candidate = input_index_path.parent.parent.parent / "eval"  # 否则往索引文件的上层目录找。
    if sibling_candidate.is_dir():  # 找到就用这个。
        return sibling_candidate  # 返回上层目录下的 eval。
    return candidate  # 都找不到就返回默认候选路径，后续会在同步时报错。


def _has_local_frozen_inputs(output_root: Path) -> bool:
    """检查验证目录下是否已经有本地冻结输入文件。"""
    required_paths = (  # 检查这四个关键文件是否都存在。
        output_root / "metrics" / "metric_table.csv",  # 指标表 CSV。
        output_root / "statistics" / "statistics_table.json",  # 统计表 JSON。
        output_root / "cases" / "selected_cases.json",  # 选例文件。
        output_root / "plotting_inputs" / "runtime_table.json",  # 运行时表。
    )
    return all(path.is_file() for path in required_paths)  # 全部存在才返回 True。


def _load_optional_json_list(path: Path, *, error_label: str) -> list[dict[str, Any]] | None:
    """读取可选的 JSON 列表文件，文件不存在时返回 None。"""
    if not path.is_file():  # 文件不存在就返回空。
        return None  # 可选文件缺失不算错误。
    payload = _load_json(path)  # 读取文件内容。
    if not isinstance(payload, list) or not payload:  # 内容必须是非空列表。
        raise ValueError(f"{error_label} must contain a non-empty list")  # 不符合就报错。
    if not isinstance(payload[0], dict):  # 每个元素必须是映射。
        raise TypeError(f"{error_label} rows must be mappings")  # 不是映射就报错。
    return payload  # 返回读取到的列表。


def _load_optional_json_mapping(path: Path, *, error_label: str) -> dict[str, Any] | None:
    """读取可选的 JSON 映射文件，文件不存在时返回 None。"""
    if not path.is_file():  # 文件不存在就返回空。
        return None  # 可选文件缺失不算错误。
    payload = _load_json(path)  # 读取文件内容。
    if not isinstance(payload, dict):  # 内容必须是映射。
        raise TypeError(f"{error_label} must contain a mapping")  # 不是映射就报错。
    return payload  # 返回读取到的映射。


def _coerce_plot_manifest(payload: Any, *, consumer_name: str) -> dict[str, Any]:
    """把绘图函数返回值规范化成统一的 manifest 字典。

    接受路径字符串、PathLike 或映射对象，统一转成带 figure_path 和 metrics 的字典。
    当绘图函数返回 skipped 结果时，原样透传。
    """
    if isinstance(payload, (str, Path)):  # 如果返回的是路径。
        return {"figure_path": str(Path(payload)), "metrics": []}  # 包装成标准 manifest。
    if not isinstance(payload, dict):  # 不是路径也不是映射就报错。
        raise TypeError(f"{consumer_name} manifest must be a mapping or path-like")  # 类型不对。
    if payload.get("skipped"):  # 绘图函数返回跳过信号时原样透传。
        return payload
    figure_path = payload.get("figure_path")  # 取出图路径。
    if not isinstance(figure_path, (str, Path)):  # 图路径必须存在。
        raise KeyError(f"{consumer_name} manifest missing figure_path")  # 缺失就报错。
    normalized_payload = dict(payload)  # 复制一份，避免修改原对象。
    normalized_payload["figure_path"] = str(Path(figure_path))  # 统一转成字符串路径。
    metrics = normalized_payload.get("metrics")  # 取出指标列表。
    if metrics is None:  # 没有指标就补空列表。
        normalized_payload["metrics"] = []  # 补上空列表。
    elif isinstance(metrics, str):  # 单个字符串指标包装成列表。
        normalized_payload["metrics"] = [metrics]  # 单字符串补成一元素列表。
    elif isinstance(metrics, Mapping):  # 映射类型的指标不接受。
        raise TypeError(f"{consumer_name} manifest metrics must be a sequence, not a mapping")  # 映射类型报错。
    elif not isinstance(metrics, list):  # 其他序列类型转成列表。
        normalized_payload["metrics"] = list(metrics)  # 转成列表。
    return normalized_payload  # 返回规范化后的 manifest。


def _coerce_case_group_payload(payload: Any, *, group_name: str) -> dict[str, str]:
    """把案例图分组返回值规范化成带 figure_path 的字典。"""
    if isinstance(payload, (str, Path)):  # 如果返回的是路径。
        return {"figure_path": str(Path(payload))}  # 包装成标准字典。
    if not isinstance(payload, dict):  # 不是路径也不是映射就报错。
        raise TypeError(f"{group_name} case figure entry must be a mapping or path-like")  # 类型不对。
    figure_path = payload.get("figure_path")  # 取出图路径。
    if not isinstance(figure_path, (str, Path)):  # 图路径必须存在。
        raise KeyError(f"{group_name} case figure entry missing figure_path")  # 缺失就报错。
    return {"figure_path": str(Path(figure_path))}  # 返回规范化后的字典。


def _coerce_success_list(payload: Any, *, field_name: str) -> list[Any]:
    """把各种形态的成功列表统一转成标准列表。"""
    if payload is None:  # 空值转成空列表。
        return []  # 没有内容就返回空列表。
    if isinstance(payload, str):  # 单字符串包装成列表。
        return [payload]  # 单字符串补成一元素列表。
    if isinstance(payload, Mapping):  # 映射类型不接受。
        raise TypeError(f"{field_name} must be a sequence, not a mapping")  # 映射类型报错。
    try:  # 尝试把其他可迭代对象转成列表。
        return list(payload)  # 转成列表。
    except TypeError as exc:  # 不可迭代就报错。
        raise TypeError(f"{field_name} must be a sequence, string, or null") from exc  # 明确类型要求。


def _resolve_path_argument(raw_value: str | None, *, default: Path, flag_name: str) -> Path:
    """把命令行路径参数规范成绝对路径。"""
    if raw_value is None:  # 没传就用默认值。
        return default.resolve()  # 默认路径转成绝对路径。
    value = str(raw_value).strip()  # 去掉空白。
    if not value:  # 空字符串不合法。
        raise ValueError(f"{flag_name} must be a non-empty path")  # 报错。
    path = Path(value)  # 转成路径对象。
    if not path.is_absolute():  # 相对路径按仓库根目录解释。
        path = (ROOT / path).resolve()  # 补成绝对路径。
    return path.resolve()  # 返回最终路径。


def _resolve_verification_paths(argv: list[str] | None = None) -> tuple[Path, Path, Path, Path, bool]:
    """解析命令行参数，返回验证所需的关键路径和输入根目录是否显式传入。"""
    parser = argparse.ArgumentParser(description="Verify high-level consumer inputs and outputs")  # 创建参数解析器。
    parser.add_argument("--input-root", default=None)  # 输入根目录，可选。
    parser.add_argument("--ground-truth-root", default=None)  # 真值根目录，可选。
    parser.add_argument("--output-root", default=None)  # 输出根目录，可选。
    args = parser.parse_args(argv)  # 解析命令行参数。

    from liquidloc.common.tee_logger import print_args
    print_args(args, "19_verify_high_level_consumers")

    input_root_explicit = args.input_root is not None  # 记录用户是否显式指定了输入根目录。
    input_root = _resolve_path_argument(  # 解析输入根目录。
        args.input_root,  # 命令行传入的值。
        default=INPUT_INDEX_PATH.parent.parent,  # 默认用索引文件的上层目录。
        flag_name="--input-root",  # 参数名，用于报错。
    )  # 输入根目录解析结束。
    ground_truth_root = _resolve_path_argument(  # 解析真值根目录。
        args.ground_truth_root,  # 命令行传入的值。
        default=GROUND_TRUTH_ROOT,  # 默认用 MILUV fixture 目录。
        flag_name="--ground-truth-root",  # 参数名，用于报错。
    )  # 真值根目录解析结束。
    output_root = _resolve_path_argument(args.output_root, default=OUTPUT_ROOT, flag_name="--output-root")  # 解析输出根目录。
    return input_root, input_root / "audits" / "prediction_index.json", ground_truth_root, output_root, input_root_explicit  # 返回解析结果。


def main(argv: list[str] | None = None) -> int:
    """脚本主入口，验证高层消费者能否正确处理冻结评估产物。"""
    argv = [] if argv is None else argv  # 确保参数列表不为 None。
    input_root, input_index_path, ground_truth_root, output_root, input_root_explicit = _resolve_verification_paths(argv)  # 解析验证路径。
    print("[19_consumer_verify] 开始 | input_root=" + str(input_root) + " output_root=" + str(output_root), flush=True)
    # 不再使用 global 修改模块级变量；所有路径通过参数传递给辅助函数。

    validated_prediction_index: list[dict[str, Any]] = []  # 保存验证通过的预测索引条目。
    synced_eval_artifacts: dict[str, str] = {}  # 保存已同步的评估产物路径。
    using_local_frozen_inputs = False  # 高层 consumer 验证必须基于本次固定输入链路，不允许静默回退到旧产物。
    try:  # 尝试验证和同步固定路线的输入。
        print("[19_consumer_verify] 正在校验固定路线输入", flush=True)
        validated_prediction_index = _validate_fixed_route_inputs(  # 验证预测索引和真值。
            index_path=input_index_path,  # 预测索引路径。
            ground_truth_root=ground_truth_root,  # 真值根目录。
        )  # 验证结束。
    except Exception as exc:  # 验证失败就返回失败摘要。
        print("[19_consumer_verify] 完成 | 返回码=<fail:route_bridge>", flush=True)
        return _fail("route_bridge", "fixed prediction_index or ground truth validation failed", str(exc))  # 报告失败。
    source_eval_root = _resolve_source_eval_root(input_root, input_index_path)  # 定位评估产物源目录。
    try:  # 尝试同步评估产物到验证目录。
        synced_eval_artifacts = _sync_fixed_eval_route(source_eval_root, output_root)  # 执行同步。
    except Exception as exc:  # 同步失败就返回失败摘要。
        print("[19_consumer_verify] 完成 | 返回码=<fail:route_bridge_sync>", flush=True)
        return _fail("route_bridge", "fixed eval route could not be synced to verification route", str(exc))  # 报告失败。

    plotting_root = _resolve_plotting_root(output_root)  # 获取绘图输出目录。
    plotting_root.mkdir(parents=True, exist_ok=True)  # 确保绘图目录存在。

    metric_path = output_root / "metrics" / "metric_table.csv"  # 指标表路径。
    statistics_path = output_root / "statistics" / "statistics_table.json"  # 统计表路径。
    cases_path = output_root / "cases" / "selected_cases.json"  # 选例路径。
    main_table_path = output_root / "plotting_inputs" / "main_table.json"  # 主表聚合视图路径。
    runtime_table_path = output_root / "plotting_inputs" / "runtime_table.json"  # 运行时表路径。
    sweep_table_path = output_root / "plotting_inputs" / "sweep_table.json"  # 扫描表路径。
    trajectory_bundle_path = output_root / "plotting_inputs" / "trajectory_bundle.json"  # 轨迹数据路径。

    for artifact_path in (metric_path, statistics_path, cases_path, main_table_path, runtime_table_path):  # 检查必须产物是否存在。
        if not artifact_path.is_file():  # 产物文件不存在就报失败。
            return _fail("artifact_check", "missing fixed verification artifact", str(artifact_path))  # 报告缺失。

    try:  # 尝试加载和验证所有冻结输入。
        metric_rows = _load_metric_rows(metric_path)  # 加载指标表。
        statistics_payload = _load_statistics_payload(statistics_path)  # 加载统计表。
        selected_cases = _load_selected_cases(cases_path)  # 加载选例。
        main_table_rows = _load_optional_json_list(main_table_path, error_label="main_table.json")  # 加载主表聚合视图。
        runtime_table = _load_runtime_table(runtime_table_path)  # 加载运行时表。
        sweep_table = _load_optional_json_list(sweep_table_path, error_label="sweep_table.json")  # 加载扫描表（可选）。
        trajectory_bundle = _load_optional_json_mapping(trajectory_bundle_path, error_label="trajectory_bundle.json")  # 加载轨迹数据（可选）。
    except Exception as exc:  # 加载失败就返回失败摘要。
        print("[19_consumer_verify] 完成 | 返回码=<fail:artifact_check>", flush=True)
        return _fail("artifact_check", "frozen inputs failed validation", str(exc))  # 报告失败。

    try:  # 尝试用 summary 构建器消费冻结输入。
        print("[19_consumer_verify] 正在构建汇总并渲染图表", flush=True)
        summary_payload = build_summary(main_table_rows, statistics_payload, selected_cases)  # 构建 summary。
        if not isinstance(summary_payload, dict):  # summary 必须是映射。
            raise TypeError("summary payload must be a mapping")  # 不是映射就报错。
        from liquidloc.common.tee_logger import print_dict
        print_dict({"input_root": str(input_root), "ground_truth_root": str(ground_truth_root), "output_root": str(output_root), "input_index_path": str(input_index_path)}, "路径配置")
    except Exception as exc:  # 构建失败就返回失败摘要。
        blocker = "summary consumer rejected frozen inputs"  # 默认阻塞原因。
        if isinstance(exc, TypeError) and str(exc) == "summary payload must be a mapping":  # 如果是返回值类型错误。
            blocker = "summary consumer returned invalid payload"  # 修改阻塞原因。
        return _fail("summary_builder", blocker, str(exc))  # 报告失败。

    try:  # 尝试用各个绘图函数消费冻结输入。
        main_table_manifest = _coerce_plot_manifest(  # 渲染主表图并规范化 manifest。
            render_main_table_figure(main_table_rows, {"figure_path": plotting_root / "main_table.png"}),  # 传入主表聚合视图和输出路径。
            consumer_name="main_table",  # 消费者名称，用于报错。
        )  # 主表图 manifest 结束。
        runtime_manifest = _coerce_plot_manifest(  # 渲染运行时图并规范化 manifest。
            render_runtime_figure(  # 调用运行时图渲染函数。
                runtime_table,  # 传入运行时表。
                {"figure_path": plotting_root / "runtime.svg", "return_manifest": True},  # 配置输出路径和返回 manifest 模式。
            ),  # 运行时图渲染结束。
            consumer_name="runtime",  # 消费者名称。
        )  # 运行时图 manifest 结束。
        calibration_manifest = _coerce_plot_manifest(  # 渲染校准图并规范化 manifest。
            render_calibration_figure(  # 调用校准图渲染函数。
                metric_rows,  # 传入指标表。
                {"figure_path": plotting_root / "calibration.svg", "return_manifest": True},  # 配置输出路径和返回 manifest 模式。
            ),  # 校准图渲染结束。
            consumer_name="calibration",  # 消费者名称。
        )  # 校准图 manifest 结束。
        sweep_manifest = None  # 扫描图 manifest 初始为空。
        if sweep_table is not None:  # 如果有扫描表数据就渲染扫描图。
            sweep_manifest = _coerce_plot_manifest(  # 渲染扫描图并规范化 manifest。
                render_sweep_figure(  # 调用扫描图渲染函数。
                    sweep_table,  # 传入扫描表。
                    {"figure_path": plotting_root / "sweep.svg", "return_manifest": True},  # 配置输出路径和返回 manifest 模式。
                ),  # 扫描图渲染结束。
                consumer_name="sweep",  # 消费者名称。
            )  # 扫描图 manifest 结束。
        trajectory_figure_path = None  # 轨迹图路径初始为空。
        if trajectory_bundle is not None:  # 如果有轨迹数据就渲染轨迹图。
            trajectory_figure_path = Path(  # 把渲染结果转成路径对象。
                render_trajectory_figure(  # 调用轨迹图渲染函数。
                    trajectory_bundle,  # 传入轨迹数据。
                    {"figure_path": plotting_root / "trajectory.png"},  # 配置输出路径。
                )  # 轨迹图渲染结束。
            )  # 路径转换结束。
        case_manifest = render_case_figures(  # 渲染案例图。
            selected_cases,  # 传入选例数据。
            {"figure_path": plotting_root / "cases.png"},  # 配置输出路径。
        )  # 案例图渲染结束。
    except Exception as exc:  # 绘图失败就返回失败摘要。
        print("[19_consumer_verify] 完成 | 返回码=<fail:plotting>", flush=True)
        return _fail("plotting", "plotting consumer rejected frozen inputs", str(exc))  # 报告失败。

    try:  # 验证绘图产物是否真的生成了文件。
        main_table_figure_path = Path(main_table_manifest["figure_path"])  # 取出主表图路径。
        runtime_figure_path = Path(runtime_manifest["figure_path"])  # 取出运行时图路径。
        calibration_figure_path = Path(calibration_manifest["figure_path"])  # 取出校准图路径。
        if not main_table_figure_path.is_file():  # 主表图文件必须存在。
            return _fail("plotting", "main table figure missing after render", str(main_table_figure_path))  # 报告缺失。
        if not runtime_figure_path.is_file():  # 运行时图文件必须存在。
            return _fail("plotting", "runtime figure missing after render", str(runtime_figure_path))  # 报告缺失。
        if not calibration_figure_path.is_file():  # 校准图文件必须存在。
            return _fail("plotting", "calibration figure missing after render", str(calibration_figure_path))  # 报告缺失。
        sweep_figure_path = None  # 扫描图路径初始为空。
        if sweep_manifest is not None:  # 如果有扫描图 manifest。
            sweep_figure_path = Path(sweep_manifest["figure_path"])  # 取出扫描图路径。
            if not sweep_figure_path.is_file():  # 扫描图文件必须存在。
                return _fail("plotting", "sweep figure missing after render", str(sweep_figure_path))  # 报告缺失。
        if trajectory_figure_path is not None and not trajectory_figure_path.is_file():  # 轨迹图文件必须存在。
            return _fail("plotting", "trajectory figure missing after render", str(trajectory_figure_path))  # 报告缺失。
        case_figure_paths = []  # 保存案例图路径列表。
        for group_name in ("main_cases", "failure_cases", "boundary_cases"):  # 逐个分组检查。
            if not isinstance(case_manifest, dict):  # 案例 manifest 必须是映射。
                return _fail("plotting", "case figure manifest must be a mapping", str(type(case_manifest).__name__))  # 报告类型错误。
            if group_name not in case_manifest:  # 分组必须存在。
                return _fail("plotting", "case figure manifest missing expected group", group_name)  # 报告缺失。
            group_payload = _coerce_case_group_payload(case_manifest[group_name], group_name=group_name)  # 规范化分组数据。
            figure_path = Path(group_payload["figure_path"])  # 取出图路径。
            if not figure_path.is_file():  # 图文件必须存在。
                return _fail("plotting", "case figure missing after render", str(figure_path))  # 报告缺失。
            case_figure_paths.append(str(figure_path))  # 记录案例图路径。
    except Exception as exc:  # 验证过程出错就返回失败摘要。
        return _fail("plotting", "plotting consumer returned invalid artifact manifest", str(exc))  # 报告失败。

    try:  # 验证 summary 返回值的结构。
        summary_stats = summary_payload.get("summary_stats", {})  # 取出统计摘要。
        if not isinstance(summary_stats, dict):  # 统计摘要必须是映射。
            raise TypeError("summary_stats must be a mapping")  # 不是映射就报错。
        case_refs = _coerce_success_list(summary_payload.get("case_refs"), field_name="summary case_refs")  # 规范化案例引用列表。
    except Exception as exc:  # 验证失败就返回失败摘要。
        print("[19_consumer_verify] 完成 | 返回码=<fail:summary_builder>", flush=True)
        return _fail("summary_builder", "summary consumer returned invalid payload", str(exc))  # 报告失败。

    print("[19_consumer_verify] 完成 | 返回码=0", flush=True)
    return _print_summary(  # 打印最终成功摘要。
        {  # 构造成功摘要对象。
            "status": "ok",  # 状态标记为成功。
            "stage": "high_level_consumers",  # 阶段名称。
            "exit_code": 0,  # 成功退出码。
            "artifacts": {  # 所有验证通过的产物路径。
                "metric_table": str(metric_path),  # 指标表路径。
                "statistics_table": str(statistics_path),  # 统计表路径。
                "selected_cases": str(cases_path),  # 选例路径。
                "main_table": str(main_table_path),  # 主表聚合视图路径。
                "runtime_table": str(runtime_table_path),  # 运行时表路径。
                "sweep_table": str(sweep_table_path),  # 扫描表路径。
                "trajectory_bundle": str(trajectory_bundle_path),  # 轨迹数据路径。
                "main_table_figure": str(main_table_figure_path),  # 主表图路径。
                "runtime_figure": str(runtime_figure_path),  # 运行时图路径。
                "calibration_figure": str(calibration_figure_path),  # 校准图路径。
                "case_figures": case_figure_paths,  # 案例图路径列表。
            },  # 产物路径结束。
            "route_bridge": {  # 路线桥接信息。
                "prediction_index": str(input_index_path),  # 预测索引路径。
                "validated_entries": len(validated_prediction_index),  # 验证通过的条目数。
                "synced_eval_artifacts": synced_eval_artifacts,  # 已同步的评估产物。
                "used_local_frozen_inputs": using_local_frozen_inputs,  # 是否使用了本地冻结输入。
            },  # 路线桥接信息结束。
            "summary": {  # summary 验证结果。
                "case_refs": case_refs,  # 案例引用列表。
                "summary_stats_keys": sorted(summary_stats.keys()),  # 统计摘要的键列表。
            },  # summary 结果结束。
            "plot_manifests": {  # 绘图 manifest 信息。
                "runtime_metrics": _coerce_success_list(  # 运行时图指标列表。
                    runtime_manifest.get("metrics"),  # 从 manifest 中取出指标。
                    field_name="runtime manifest metrics",  # 字段名，用于报错。
                ),  # 运行时指标列表结束。
                "calibration_metrics": _coerce_success_list(  # 校准图指标列表。
                    calibration_manifest.get("metrics"),  # 从 manifest 中取出指标。
                    field_name="calibration manifest metrics",  # 字段名，用于报错。
                ),  # 校准指标列表结束。
                "sweep": sweep_manifest,  # 扫描图 manifest。
                "trajectory": None if trajectory_figure_path is None else {"figure_path": str(trajectory_figure_path)},  # 轨迹图信息。
            },  # 绘图 manifest 结束。
            "eval_stage_name": EVAL_STAGE_NAME,  # 评估阶段名称。
        }  # 成功摘要对象结束。
    )  # 打印结束。


if __name__ == "__main__":  # 只有直接执行脚本时才走这里。
    raise SystemExit(main(sys.argv[1:]))  # 用 main 的返回码结束进程。
