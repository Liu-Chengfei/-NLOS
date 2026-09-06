"""脚本：运行最小的公开 benchmark 冒烟流程。"""

from __future__ import annotations  # 允许在类型注解里直接引用后续定义的类型名。

import argparse  # 负责解析命令行参数。
import json  # 负责输出结果摘要和就绪报告。
import sys  # 负责调整解释器导入路径。
from pathlib import Path  # 负责表示和规整文件路径。

ROOT = Path(__file__).resolve().parents[1]  # 从脚本位置反推仓库根目录。
SRC = ROOT / "src"  # 保存源码目录，后面要加到导入路径里。
if str(SRC) not in sys.path:  # 如果源码目录还没有进入搜索路径。
    sys.path.insert(0, str(SRC))  # 把源码目录插到最前面，优先导入仓库代码。

from liquidloc.common.config_utils import load_yaml_config  # 读取 YAML 配置。
from liquidloc.common.io_utils import dumps_json_text, write_json  # 严格 JSON 读写，拒绝 NaN/Infinity。
from liquidloc.dataio.registry.public_dataset_registry import (  # 统一公开数据集注册表访问。
    get_dataset_entry,
    load_public_dataset_registry,
    normalize_public_dataset_name,
    resolve_public_eval_seq_ids,
)
from liquidloc.dataio.readers.miluv_reader import inspect_miluv_raw_readiness  # 检查 MILUV 原始数据是否就绪。
from liquidloc.dataio.readers.ntu_viral_reader import inspect_ntu_viral_raw_readiness  # 检查 NTU VIRAL 原始数据是否就绪。
from liquidloc.pipelines.eval_pipeline import EvalPipeline  # 评估流水线入口。
from liquidloc.pipelines.public_benchmark_pipeline import PublicBenchmarkPipeline  # 公开 benchmark 流水线入口。
from liquidloc.protocol.experiment_gates import (  # 读取协议定义的官方公开 benchmark 白名单与论文级就绪检查。
    check_paper_grade_readiness,
    get_public_benchmark_allowed_datasets,
)

_SUPPORTED_PUBLIC_DATASETS = None


def _get_supported_public_datasets():
    global _SUPPORTED_PUBLIC_DATASETS
    if _SUPPORTED_PUBLIC_DATASETS is None:
        _SUPPORTED_PUBLIC_DATASETS = get_public_benchmark_allowed_datasets()
    return _SUPPORTED_PUBLIC_DATASETS


def _resolve_project_relative_path(path_value: str | None) -> str | None:  # 把命令行路径统一解析成仓库根目录下的绝对字符串。
    """把命令行传入的路径解析成项目根目录下的绝对字符串。"""
    if path_value is None:  # 没传值就保持空。
        return None  # 后续由默认分支决定真正路径。
    if str(path_value).strip() == "":  # 空字符串不算有效路径。
        raise ValueError("--raw-root must be a non-empty path")  # 直接拒绝空路径输入。
    path = Path(path_value)  # 把输入包装成路径对象。
    if not path.is_absolute():  # 相对路径按仓库根目录解释。
        path = (ROOT / path).resolve()  # 拼成绝对路径，避免依赖当前工作目录。
    else:  # 如果本来就是绝对路径。
        path = path.resolve()  # 也统一规整一次。
    return str(path)  # 返回字符串，便于后续 JSON 和流水线参数使用。


def _resolve_output_root(output_root: str | Path | None, default_relative: str) -> Path:  # 解析输出目录。
    """解析输出目录。"""
    if output_root is None:  # 没传就用默认位置。
        return (ROOT / default_relative).resolve()  # 默认路径也要规整成绝对路径。
    if str(output_root).strip() == "":  # 显式空字符串不能当作当前目录吞掉。
        raise ValueError("--output-root must be a non-empty path")  # 直接阻断错误参数，避免产物静默落到仓库根目录。
    path = Path(output_root)  # 把输入转成路径对象。
    if not path.is_absolute():  # 相对路径按仓库根目录解释。
        return (ROOT / path).resolve()  # 返回补全后的绝对路径。
    return path.resolve()  # 已经是绝对路径就直接规整返回。


def _inspect_public_raw_readiness(dataset_name: str, raw_root: str) -> dict[str, object]:  # 统一封装公开数据集原始数据就绪检查。
    """检查公开数据集原始数据是否就绪。"""
    if dataset_name == "miluv":  # MILUV 走对应检查器。
        return inspect_miluv_raw_readiness(raw_root)  # 直接返回 MILUV 的就绪报告。
    if dataset_name == "ntu_viral":  # NTU VIRAL 走对应检查器。
        return inspect_ntu_viral_raw_readiness(raw_root)  # 直接返回 NTU VIRAL 的就绪报告。
    return {  # 不支持的数据集直接返回失败报告。
        "dataset_name": dataset_name,  # 报告里要保留数据集名。
        "raw_root": str(raw_root),  # 报告里要保留原始目录。
        "status": "not_ready",  # 默认判定为未就绪。
        "gate_action": "blocked",  # 门控动作是阻断。
        "reasons": ["unsupported_public_dataset"],  # 说明为什么被阻断。
        "sequence_count": 0,  # 没有可处理序列。
        "ready_sequence_count": 0,  # 也没有已就绪序列。
    }  # 报告对象结束。


def _emit_readiness_block(public_output_root: Path, dataset_name: str, raw_root: str, readiness_report: dict[str, object]) -> int:
    """把数据就绪阻断统一落盘并打印成命令行摘要。"""
    report_path = public_output_root / f"{dataset_name}_readiness.json"  # 就绪度报告路径。
    write_json(report_path, readiness_report)  # 写出严格 JSON 就绪度报告。
    print(
        dumps_json_text(
            {
                "status": "blocked",
                "stage": "data_readiness",
                "dataset_name": dataset_name,
                "raw_root": raw_root,
                "report_path": str(report_path),
                "reasons": readiness_report.get("reasons") or [],
            }
        )
    )
    return 2


def _auto_fetch_public_dataset(dataset_name: str, raw_root: str) -> dict:
    """数据缺失时自动调用 16 号脚本完成 fetch+convert+verify。

    设计目的：让用户跑 18 号脚本时不必先手动执行下载，实现"零准备"体验。
    16 脚本使用 registry 中的 official_root / landed_raw_root 决定下载和落地位置。
    返回值含 "landed_raw_root" 字段，调用方可据此更新后续就绪检查路径。
    """
    import subprocess  # 局部导入，避免顶层依赖。

    download_script = ROOT / "scripts" / "16_download_public_datasets.py"  # 16 号脚本路径。
    if not download_script.is_file():  # 16 脚本不存在时无法自动下载。
        return {"status": "unavailable", "reasons": ["download_script_missing"]}

    # 解析 registry 获取真实落地路径，fetch 成功后需将 raw_root 指向此处。
    registry_cfg = load_public_dataset_registry()
    dataset_entry = get_dataset_entry(dataset_name, registry_cfg)
    landed_raw_root = str(
        Path(dataset_entry.get("landed_raw_root", ROOT / "data" / "raw" / dataset_name)).resolve()
    )

    cmd = [
        sys.executable,
        str(download_script),
        "--dataset-name",
        dataset_name,
        "--mode",
        "full",
    ]
    print("[18_pub_benchmarks] 触发自动下载+解包 | cmd=" + " ".join(cmd), flush=True)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            check=False,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "reasons": [f"subprocess_error: {exc}"],
            "landed_raw_root": landed_raw_root,
        }

    return {
        "status": "ready" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "landed_raw_root": landed_raw_root,
    }


def _resolve_public_seq_ids(dataset_name: str, experiment_cfg: dict[str, object], mode: str) -> list[str]:
    """Resolve public benchmark sequence IDs from the frozen registry contract plus mode scale."""
    return resolve_public_eval_seq_ids(dataset_name, dict(experiment_cfg), mode)


def main(argv: list[str] | None = None) -> int:  # 脚本主入口，串起配置、就绪检查和评估。
    """脚本主入口，检查公开原始数据并运行公开 benchmark。"""
    parser = argparse.ArgumentParser(description="Run minimal public benchmark smoke")  # 创建参数解析器。
    parser.add_argument("--dataset-name", default="miluv")  # 数据集名，默认是 MILUV。
    parser.add_argument("--raw-root", default=None)  # 原始数据目录覆盖值。
    parser.add_argument("--output-root", default=None)  # 输出目录覆盖值。
    parser.add_argument("--mode", choices=("quick", "full", "paper"), default="quick")  # 运行规模模式（paper=论文级，需要 paper_full_ready=true）。
    parser.add_argument("--no-fetch", action="store_true", default=False)  # 跳过自动下载，仅在已就绪数据时运行。
    args = parser.parse_args(argv)  # 解析命令行参数。

    from liquidloc.common.tee_logger import print_args, print_dict
    print_args(args, "18_run_public_benchmarks")

    print("[18_pub_benchmarks] 开始 | dataset_name=" + str(args.dataset_name) + " mode=" + str(args.mode), flush=True)

    dataset_name = normalize_public_dataset_name(args.dataset_name)  # 当前要跑的数据集名。
    public_output_root = _resolve_output_root(args.output_root, "outputs/public_script_smoke")  # 解析公开 benchmark 输出目录。
    public_output_root.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
    raw_root = args.raw_root  # 先拿命令行原始值。
    if dataset_name not in _get_supported_public_datasets():  # 未支持的数据集要先于任何配置加载被阻断。
        if raw_root is None:
            raw_root = str((ROOT / "tests" / "fixtures" / "datasets" / dataset_name).resolve())
        else:
            raw_root = _resolve_project_relative_path(raw_root)
        readiness_report = _inspect_public_raw_readiness(dataset_name, raw_root)
        print("[18_pub_benchmarks] 完成 | 返回码=<emit_readiness_block:unsupported>", flush=True)
        return _emit_readiness_block(public_output_root, dataset_name, raw_root, readiness_report)
    dataset_cfg = load_yaml_config(ROOT / "configs" / "datasets" / f"{dataset_name}.yaml")  # 读取数据集配置。
    print_dict(dataset_cfg, "数据集配置")
    default_raw_root = ROOT / "tests" / "fixtures" / "datasets" / dataset_name  # 默认 fixture 数据目录。
    if raw_root is None:  # 如果命令行没显式指定原始目录。
        if default_raw_root.is_dir():  # 如果 fixture 目录存在。
            raw_root = str(default_raw_root)  # 优先使用 fixture。
        else:  # 否则回退到配置里的 raw_root。
            raw_root = str(ROOT / dataset_cfg["raw_root"])  # 按仓库根目录补全配置路径。
    else:  # 如果命令行显式传了 raw_root。
        raw_root = _resolve_project_relative_path(raw_root)  # 解析成绝对路径字符串。
    experiment_cfg = load_yaml_config(ROOT / "configs" / "experiments" / f"e7_{dataset_name}.yaml")  # 读取实验配置。
    print_dict(experiment_cfg, "实验配置 (e7)")
    field_mapping = dataset_cfg["field_mapping"]  # 取出字段映射，供公开 benchmark 和评估链路共用。
    readiness_report = _inspect_public_raw_readiness(dataset_name, raw_root)  # 检查原始数据是否就绪。
    if dataset_name in _get_supported_public_datasets() and readiness_report["status"] != "ready":  # 如果支持但数据还没准备好。
        # 零准备体验：默认自动触发下载+解包，让"直接跑"就能出结果；
        # 除非用户显式传入 --no-fetch 要求只在已就绪数据时运行。
        if not getattr(args, "no_fetch", False):
            fetch_status = _auto_fetch_public_dataset(dataset_name, raw_root)
            print_dict(fetch_status, "自动下载+解包结果")
            if fetch_status.get("status") == "ready":
                # 下载成功：把 raw_root 重定向到真实落地目录，再重新做就绪检查。
                landed = fetch_status.get("landed_raw_root")
                if landed:
                    raw_root = str(Path(landed).resolve())
                readiness_report = _inspect_public_raw_readiness(dataset_name, raw_root)  # 重新检查就绪状态。
        if readiness_report["status"] != "ready":  # 自动拉取仍不就绪则阻断。
            print("[18_pub_benchmarks] 完成 | 返回码=<emit_readiness_block:not_ready>", flush=True)
            return _emit_readiness_block(public_output_root, dataset_name, raw_root, readiness_report)  # 数据不就绪时返回阻断码。

    # 论文级模式：仅当 paper_full_ready=true 且 release_tier 为 full_ready/paper_ready 时放行。
    if args.mode == "paper":
        paper_readiness = check_paper_grade_readiness(dataset_name)
        if paper_readiness["status"] != "ready":
            print("[18_pub_benchmarks] 完成 | 返回码=<emit_readiness_block:paper_not_ready>", flush=True)
            return _emit_readiness_block(
                public_output_root,
                dataset_name,
                raw_root,
                {"dataset_name": dataset_name, "status": "not_ready", "reasons": paper_readiness["reasons"]},
            )

    seq_ids = _resolve_public_seq_ids(dataset_name, experiment_cfg, args.mode)  # 按冻结评测合同和 quick/full 规模规则派生序列。
    print("[18_pub_benchmarks] 正在运行公开 benchmark 流水线 | seq_count=" + str(len(seq_ids)), flush=True)
    public_payload = {  # 组装流水线输入对象。
        "dataset_name": dataset_name,  # 数据集名。
        "raw_root": raw_root,  # 原始数据目录。
        "field_mapping": field_mapping,  # 字段映射。
        "seq_ids": seq_ids,  # 要跑的序列列表。
        "methods": list(experiment_cfg.get("methods") or []),  # 实验配置里的方法列表。
        "mode": args.mode,  # 运行模式。
        "output_root": str(public_output_root),  # 输出目录。
    }  # 输入对象结束。
    print_dict(public_payload, "公开基准 payload")
    public_result = PublicBenchmarkPipeline().run(public_payload)  # 执行公开 benchmark 流水线。
    prediction_bundles = list(public_result.metadata.get("prediction_bundles") or [])  # 取出评估阶段要用的预测 bundle。
    if not prediction_bundles:  # 如果没有输出 bundle。
        raise ValueError("Public benchmark pipeline must produce prediction_bundles for eval handoff")  # 直接报错，防止评估链断掉。

    eval_output_root = public_output_root / "eval"  # 单独给评估链建一个输出目录。
    eval_output_root.mkdir(parents=True, exist_ok=True)  # 确保目录存在。
    print("[18_pub_benchmarks] 正在运行评估流水线", flush=True)
    eval_payload = {  # 组装评估输入。
        "prediction_bundles": prediction_bundles,  # 预测 bundle 作为评估输入。
        "ground_truth_root": raw_root,  # 真值根目录。
        "mode": args.mode,  # 让 eval gate 继承同一次 public run 的 quick/full 口径。
        "output_root": str(eval_output_root),  # 评估输出目录。
    }  # 评估输入结束。
    print_dict(eval_payload, "评估执行 payload")
    eval_result = EvalPipeline().run(eval_payload)  # 执行评估流水线。
    print(  # 打印总摘要，方便查看闭环结果。
        dumps_json_text(  # 序列化成严格 JSON。
            {  # 摘要对象开始。
                "stage_name": "public_benchmark_closed_loop",  # 闭环阶段名。
                "public_stage_name": public_result.stage_name,  # 公开 benchmark 阶段名。
                "eval_stage_name": eval_result.stage_name,  # 评估阶段名。
                "artifacts": list(public_result.artifacts) + list(eval_result.artifacts),  # 合并两个阶段的产物列表。
                "report": public_result.metadata["public_benchmark_report"],  # 公开 benchmark 报告。
                "eval_audit": eval_result.metadata["audit_report"],  # 评估审计报告。
            }  # 摘要对象结束。
        )  # JSON 序列化结束。
    )  # 打印结束。
    print("[18_pub_benchmarks] 完成 | 返回码=0", flush=True)
    return 0  # 正常退出。


if __name__ == "__main__":  # 只有被直接执行时才进入这个分支。
    raise SystemExit(main())  # 用 main 的返回码结束进程。
