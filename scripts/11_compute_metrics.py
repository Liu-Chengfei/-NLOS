"""脚本：从 prediction bundle 和 ground-truth bundle 计算一个小型指标表。

这个脚本只做一件事：把已经准备好的预测结果和真值结果送进指标计算器，
再把输出整理成一个稳定的 JSON 文件。它面向冒烟验证、脚本串联和后续图表
消费，不负责训练、不负责采样，也不负责解释指标本身的科学含义。
"""

from __future__ import annotations  # 允许在类型注解里直接引用当前模块里的类型名。

import argparse  # 负责解析命令行参数。
import json  # 负责读取和写出 JSON。
import sys  # 负责调整解释器导入路径。
from collections.abc import Mapping, Sequence  # 用来判断映射和序列类型。
from pathlib import Path  # 用来表示和规整文件路径。
from typing import Any  # 用来标注任意 JSON 风格对象。

ROOT = Path(__file__).resolve().parents[1]  # 从脚本文件位置反推仓库根目录。
SRC = ROOT / "src"  # 显式保存源码目录，后面要把它加到导入路径里。
if str(SRC) not in sys.path:  # 如果源码目录还没有被解释器搜索到。
    sys.path.insert(0, str(SRC))  # 把源码目录插到最前面，保证优先导入仓库代码。

from liquidloc.metrics.metric_runner import compute_metrics  # 真正执行指标计算的入口函数。
from liquidloc.protocol.metric_schema import get_metric_order  # 读取指标顺序，用于输出摘要中的计数。


def _resolve_path(raw_value: str | None, *, flag_name: str, default: Path | None = None) -> Path:
    """把命令行传入的路径统一规整成绝对路径。

    参数:
        raw_value: 命令行读到的原始值，可能是 `None`、空字符串、相对路径或绝对路径。
        flag_name: 出错时用于提示用户的参数名，方便快速定位是哪一个开关传错了。
        default: 当 `raw_value` 没提供时使用的默认路径；如果这里也没有值就报错。

    返回:
        规整后的绝对路径。

    失败条件:
        传入空值且没有默认值，或者传入空白字符串时，都会直接抛 `ValueError`。
    """
    if raw_value is None:  # 用户没传值时，优先看是否有默认路径可用。
        if default is None:  # 默认路径也没有，说明这个参数本来就必须显式提供。
            raise ValueError(f"{flag_name} must be a non-empty path")  # 明确告诉调用者这个参数不能为空。
        return default.resolve()  # 默认路径也转成绝对路径，避免后续出现相对路径歧义。
    value = str(raw_value).strip()  # 先转字符串，再去掉首尾空白，避免空格伪装成有效路径。
    if not value:  # 空字符串和没传值在这里都视为无效输入。
        raise ValueError(f"{flag_name} must be a non-empty path")  # 这里不猜测用户意图，直接拒绝。
    path = Path(value)  # 把字符串包装成 Path 对象，方便判断相对/绝对路径。
    if not path.is_absolute():  # 相对路径要按仓库根目录解释，不能依赖当前工作目录。
        path = (ROOT / path).resolve()  # 先拼到仓库根目录，再统一规整为绝对路径。
    return path.resolve()  # 最后再规整一次，确保返回值一定是标准绝对路径。


def _read_json(path: Path) -> Any:
    """读取 JSON 文件并返回反序列化后的 Python 对象。

    参数:
        path: 需要读取的 JSON 文件路径。

    返回:
        解析后的 Python 对象，通常是字典、列表或者它们的嵌套结构。
    """
    from liquidloc.common.io_utils import read_json
    return read_json(path)  # 先按 UTF-8 读取文本，再交给严格 JSON 解析器反序列化。


def _normalize_bundle_group(bundle_obj: Any) -> list[dict[str, Any]]:
    """把单个 bundle 或 bundle 列表统一成标准列表形式。

    这个函数的作用，是把外部可能传来的两种输入形态统一掉：
    1. 单个 bundle 对象。
    2. bundle 对象列表。

    统一之后，后面的共享元信息提取和指标表组装就能按同一套逻辑处理。
    """
    if isinstance(bundle_obj, Mapping):  # 如果传进来的是单个对象，就把它包装成长度为 1 的列表。
        return [dict(bundle_obj)]  # 用 `dict()` 拷贝一份，避免后续修改影响原对象。
    if isinstance(bundle_obj, Sequence) and not isinstance(bundle_obj, (str, bytes)):  # 序列可以是列表/元组，但字符串不算。
        normalized_group = []  # 这里保存标准化后的 bundle 列表。
        for index, item in enumerate(bundle_obj):  # 逐个检查每一个元素，避免混入非法项。
            if not isinstance(item, Mapping):  # 每个元素都必须是映射对象。
                raise TypeError(f"bundle entry {index} must be an object")  # 错误信息要指出具体是哪个元素坏了。
            normalized_group.append(dict(item))  # 逐项拷贝成普通字典，方便后续统一访问。
        if not normalized_group:  # 空列表没有任何可计算内容。
            raise ValueError("bundle payload must be non-empty")  # 空输入直接拒绝，避免后面算出假结果。
        return normalized_group  # 返回已经规整好的标准列表。
    raise TypeError("bundle payload must be an object or a list of objects")  # 既不是对象也不是对象列表时，直接报错。


def _resolve_shared_metadata(bundle_group: list[dict[str, Any]]) -> dict[str, Any]:
    """从一组 bundle 中提取所有条目都一致的共享元信息。

    这里不会把所有字段都无脑带上，而是只保留那些每个 bundle 都出现、
    并且值完全相同的字段。这样写进输出表时，才不会把矛盾信息混进去。
    """
    metadata: dict[str, Any] = {}  # 这里保存筛出来的共享字段。
    for field_name in ("seq_id", "scene_id", "method_name", "task_id", "repeat_id"):  # 这些字段是当前脚本关心的共用标签。
        field_values = []  # 用来收集每个 bundle 中这个字段的值。
        for bundle in bundle_group:  # 逐个 bundle 检查这个字段是否存在。
            if field_name not in bundle or bundle[field_name] is None:  # 任一条目缺字段或值为空，都不算全组共享。
                field_values = []  # 清空已经收集到的值，避免把“部分共享”误当成“全局共享”。
                break  # 这个字段已经不满足“所有条目都共享”，可以直接停。
            field_values.append(bundle[field_name])  # 记录字段值，后面统一比较。
        if len(field_values) != len(bundle_group):  # 必须每个 bundle 都提供了这个字段，才允许继续。
            continue  # 因为没有共同信息可提取。
        first_value = field_values[0]  # 把第一个值当作基准值。
        if all(value == first_value for value in field_values[1:]):  # 只有所有条目都提供且值完全一致时才算共享。
            metadata[field_name] = first_value  # 只有真正共享的信息才写进去。
    return metadata  # 返回提取好的共享元信息字典。


def _build_metric_row(metric_values: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    """把指标值和共享元信息合成一行表格。

    这个函数只负责表格拼装，不修改任何指标值本身。
    输出是一个普通字典，其中指标名作为列名，共享元信息作为辅助列。
    """
    row: dict[str, Any] = {}  # 先准备一个空行，后面逐列填充。
    for metric_name, metric_value in metric_values.items():  # 逐个把指标写进结果行。
        row[str(metric_name)] = metric_value  # 列名统一转成字符串，避免外部传入非字符串键。
    for field_name in ("seq_id", "scene_id", "method_name", "task_id", "repeat_id"):  # 再按固定顺序补共享元信息。
        if field_name in metadata:  # 只有确实提取出来的字段才写入。
            row[field_name] = metadata[field_name]  # 把共享标签放进同一行，方便下游消费。
    return row  # 返回最终拼好的单行指标表。


def _default_output_path() -> Path:
    """返回没有显式指定输出路径时使用的默认位置。

    这个默认路径专门给脚本冒烟场景准备，避免用户什么都不传时写到奇怪目录。
    """
    return ROOT / "outputs" / "script_smoke" / "metric_table.json"  # 默认落到脚本冒烟输出目录。


def _print_summary(payload: dict[str, Any]) -> int:
    """打印结构化摘要并返回退出码。"""
    from liquidloc.common.io_utils import dumps_json_text
    print(dumps_json_text(payload, indent=None))  # 统一输出结构化 JSON，便于上游脚本消费。
    return int(payload["exit_code"])  # 返回约定好的退出码。


def _safe_write_metric_table(output_path: Path, payload: Mapping[str, Any]) -> tuple[bool, str | None, str | None]:
    """尽力写出指标表文件，失败时返回错误类型和详情。"""
    try:
        output_path.write_text(  # 将结构化指标表落盘成 JSON。
            __import__("liquidloc.common.io_utils", fromlist=["dumps_json_text"]).dumps_json_text(payload),  # 写盘前统一做严格 JSON 序列化。
            encoding="utf-8",  # 写文件时统一使用 UTF-8。
        )
    except OSError as exc:
        return False, type(exc).__name__, str(exc)
    return True, None, None


def main(argv: list[str] | None = None) -> int:
    """脚本主入口，读取预测和真值后输出指标表。

    执行顺序固定为：
    1. 解析命令行。
    2. 规整输入输出路径。
    3. 读取预测 bundle 和真值 bundle。
    4. 调用指标计算入口。
    5. 拼出最终 JSON 并写盘。

    返回值:
        0 表示成功结束。
    """
    print("[11_metrics] 开始 | 从预测/真值 bundle 计算指标表", flush=True)
    parser = argparse.ArgumentParser(description="Compute a compact metric table")  # 创建命令行参数解析器。
    parser.add_argument("--prediction-bundle", required=True)  # 预测 bundle 路径，必须提供。
    parser.add_argument("--gt-bundle", required=True)  # 真值 bundle 路径，必须提供。
    parser.add_argument("--failure-threshold", type=float, default=None)  # 失败阈值默认留空，让指标内核跟随实验协议默认值。
    parser.add_argument("--output-path", default=None)  # 输出路径，可不传。
    parser.add_argument(
        "--protocol",
        default=None,
        help=(
            "可选实验协议 YAML 路径；§9 字段 (cold_start_offset_s / cold_start_global_enforced / "
            "t_eff_min_s / takeoff_landing_policy / floor_transition_policy) 从协议 scene_scale 段读取。"
            "未传时使用 metric_runner 内部默认（不动用冷启动与制度段守门）。"
        ),
    )
    args = parser.parse_args(argv)  # 真正读取命令行参数。

    from liquidloc.common.tee_logger import print_args, print_dict
    print_args(args, "11_compute_metrics")

    try:
        print("[11_metrics] 解析路径并读取 bundle", flush=True)
        prediction_path = _resolve_path(args.prediction_bundle, flag_name="--prediction-bundle")  # 规范预测输入路径。
        gt_path = _resolve_path(args.gt_bundle, flag_name="--gt-bundle")  # 规范真值输入路径。
        output_path = _resolve_path(args.output_path, flag_name="--output-path", default=_default_output_path())  # 没传时用默认目录。

        prediction_bundle = _read_json(prediction_path)  # 读取预测 bundle 的原始 JSON 内容。
        gt_bundle = _read_json(gt_path)  # 读取真值 bundle 的原始 JSON 内容。
        print_dict({"prediction_path": str(prediction_path), "gt_path": str(gt_path), "output_path": str(output_path), "failure_threshold": args.failure_threshold}, "路径与阈值配置")
        prediction_group = _normalize_bundle_group(prediction_bundle)  # 把预测 bundle 统一成列表，方便后面提共享标签。
        print("[11_metrics] 计算指标", flush=True)
        # §9 协议字段: 解析 YAML 协议 (可选), 让 metric_runner 应用 cold_start / §9.1 t_eff_min_s /
        # takeoff_landing_policy / floor_transition_policy 等守门与制度段处理.
        protocol_cfg = None
        if args.protocol is not None:
            protocol_path = _resolve_path(args.protocol, flag_name="--protocol")
            from liquidloc.protocol.experiment_gates import load_experiment_protocol
            protocol_cfg = load_experiment_protocol(protocol_path=str(protocol_path))
        metric_values, support_report = compute_metrics(  # 调用核心指标计算器。
            prediction_bundle,  # 这里传入原始预测 bundle，由指标计算器自己解释。
            gt_bundle,  # 这里传入原始真值 bundle，与预测输入配对计算。
            failure_threshold=(float(args.failure_threshold) if args.failure_threshold is not None else None),  # 仅在显式传参时覆盖协议阈值，否则沿用 compute_metrics 内部的协议默认。
            return_support=True,  # 要求额外返回支持信息，方便后续检查。
            protocol_cfg=protocol_cfg,  # §9 协议字段 (cold_start_offset_s / t_eff_min_s / takeoff_landing_policy / floor_transition_policy) 传入内核, 让 metric_runner 按 §9 守门与制度段处理.
        )  # 指标计算调用结束。
        metric_row = _build_metric_row(metric_values, _resolve_shared_metadata(prediction_group))  # 把指标和共享标签拼成一行。
        payload = {  # 准备最终写盘对象。
            "metric_table": [metric_row],  # 指标表始终以单行列表形式输出，保持表结构一致。
            "support_report": dict(support_report) if isinstance(support_report, Mapping) else support_report,  # 如果是映射就拷成普通字典，其他类型原样保留。
        }  # 输出对象构造结束。

        output_path.parent.mkdir(parents=True, exist_ok=True)  # 先确保输出目录存在，避免写文件时报错。
    except Exception as exc:
        _rc = _print_summary(
            {
                "status": "failed",
                "stage": "compute_metrics",
                "error_type": type(exc).__name__,
                "detail": str(exc),
                "exit_code": 1,
            }
        )
        print(f"[11_metrics] 完成 | 返回码={_rc}", flush=True)
        return _rc

    print("[11_metrics] 写入指标表", flush=True)
    write_ok, error_type, error_detail = _safe_write_metric_table(output_path, payload)  # 尽力写出指标表 JSON。
    if not write_ok:
        _rc = _print_summary(
            {
                "status": "failed",
                "stage": "compute_metrics",
                "output_path": str(output_path.resolve()),
                "error_type": error_type,
                "detail": error_detail,
                "exit_code": 1,
            }
        )
        print(f"[11_metrics] 完成 | 返回码={_rc}", flush=True)
        return _rc
    _rc = _print_summary(
        {
            "status": "ok",
            "stage": "compute_metrics",
            "output_path": str(output_path.resolve()),
            "metric_count": len(get_metric_order()),
            "exit_code": 0,
        }
    )
    print(f"[11_metrics] 完成 | 返回码={_rc}", flush=True)
    return _rc


if __name__ == "__main__":  # 只有被直接执行时才进入这个分支。
    raise SystemExit(main())  # 用 `main()` 的返回码作为进程退出码。
