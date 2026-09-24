"""把 UTIL 原始数据准备成最小稳定映射包。"""

from __future__ import annotations  # 允许在类型注解里直接引用后面定义的类型。

import argparse  # 解析命令行参数。
import sys  # 调整导入路径。
from pathlib import Path  # 处理路径。
from typing import Any  # 标注任意 JSON 结构。

ROOT = Path(__file__).resolve().parents[1]  # 仓库根目录。
SRC = ROOT / "src"  # 项目源码目录。
if str(SRC) not in sys.path:  # 如果源码目录还没进导入路径。
    sys.path.insert(0, str(SRC))  # 把源码目录插到前面。

from liquidloc.common.config_utils import load_yaml_config  # 读取 YAML 配置。
from liquidloc.common.io_utils import dumps_json_text, write_json  # 严格 JSON helper。
from liquidloc.dataio.adapters.field_mapper import map_external_fields  # 把外部字段映射成内部字段。
from liquidloc.dataio.readers.util_reader import read_util_sequence  # 读取 UTIL 单个序列。


def _config_raw_root_anchor(config_path: Path) -> Path:  # 判断 raw_root 相对路径应该以哪个目录为锚点。
    """决定配置里相对 raw_root 应该相对于哪里解释。"""
    repo_dataset_cfg_dir = (ROOT / "configs" / "datasets").resolve()  # 仓库内数据集配置目录，用来判断配置文件是不是仓库自带的那一份。
    if config_path.resolve().is_relative_to(repo_dataset_cfg_dir):  # 如果配置文件就在仓库配置目录里。
        return ROOT  # 这时相对 raw_root 按仓库根目录解释，和仓库内其他数据集配置保持一致。
    return config_path.parent  # 否则按配置文件所在目录解释，方便外部复制一份配置后仍然能用。


def _resolve_config_raw_root(config_path: Path, util_cfg: dict[str, Any]) -> Path:  # 把配置里的 raw_root 统一解析成绝对路径。
    """把 UTIL 配置里的 raw_root 解析成绝对路径。"""
    raw_root_value = util_cfg.get("raw_root")  # 先从配置里把 raw_root 取出来。
    if isinstance(raw_root_value, Path):  # 如果它本来就是 Path。
        raw_root_text = str(raw_root_value)  # 统一转成字符串，后面走同一套逻辑。
    elif isinstance(raw_root_value, str):  # 如果它本来就是字符串。
        raw_root_text = raw_root_value  # 直接保留这个字符串。
    else:  # 其他类型都不接受。
        raise ValueError("UTIL config must define a non-empty raw_root")  # 这里直接报错，防止配置结构漂移。
    if not raw_root_text.strip():  # 空字符串不算有效路径。
        raise ValueError("UTIL config must define a non-empty raw_root")  # 空值和缺值在这里都拒绝。
    raw_root = Path(raw_root_text)  # 把字符串转成路径对象，方便判断是否绝对路径。
    if not raw_root.is_absolute():  # 相对路径按锚点目录解释。
        raw_root = _config_raw_root_anchor(config_path) / raw_root  # 通过锚点目录补成绝对路径。
    return raw_root.resolve()  # 返回最终绝对路径，避免后面再出现相对路径语义。


def _resolve_raw_root(config_path: Path, raw_root_override: str | None) -> Path:  # 优先使用命令行，否则回退到配置文件。
    """优先使用命令行 raw_root，否则从配置里读取。"""
    if raw_root_override is not None:  # 如果命令行显式给了 raw_root。
        if not raw_root_override.strip():  # 空字符串不允许。
            raise ValueError("UTIL --raw-root must be a non-empty path")  # 这里直接报错，避免把空字符串当成有效目录。
        return Path(raw_root_override).resolve()  # 直接返回命令行路径，命令行优先级高于配置。
    config_path = config_path.resolve()  # 先把配置路径转绝对路径，避免后面受当前工作目录影响。
    util_cfg = load_yaml_config(config_path)  # 读取 UTIL 配置，拿到原始数据目录和字段映射。
    return _resolve_config_raw_root(config_path, util_cfg)  # 由配置继续解析 raw_root。


def _resolve_seq_ids(raw_root: Path, seq_ids_arg: str | None) -> list[str]:  # 决定这次要准备哪些 UTIL 序列。
    """确定要准备哪些序列。"""
    if seq_ids_arg is not None:  # 如果命令行显式传了序列列表。
        seq_ids = list(dict.fromkeys(item.strip() for item in seq_ids_arg.split(",") if item.strip()))  # 先去空白，再去重，保持命令行顺序。
        if seq_ids:  # 非空才接受。
            return seq_ids  # 直接用命令行指定的序列。
        raise ValueError("UTIL --seq-ids must be a non-empty comma-separated list")  # 空列表不接受。
    if not raw_root.is_dir():  # 原始目录不存在时不能继续。
        raise FileNotFoundError(f"UTIL raw root not found: {raw_root}")  # 这里明确告诉用户缺的是哪个目录。
    seq_ids = sorted(path.name for path in raw_root.iterdir() if path.is_dir())  # 把 raw_root 下的子目录名当作序列 ID。
    if not seq_ids:  # 没有任何序列也不行。
        raise ValueError(f"No UTIL sequences found under: {raw_root}")  # 没有序列就说明原始目录还没准备好。
    return seq_ids  # 返回最终决定的序列列表。


def _print_summary(payload: dict[str, Any]) -> int:
    """打印结构化摘要并返回退出码。"""
    print(dumps_json_text(payload))
    return int(payload["exit_code"])


def _fail(stage: str, detail: str, *, error_type: str) -> int:
    """打印结构化失败摘要。"""
    return _print_summary(
        {
            "status": "failed",
            "stage": stage,
            "error_type": error_type,
            "detail": detail,
            "exit_code": 1,
        }
    )


def main(argv: list[str] | None = None) -> int:  # 脚本主入口，串起配置、读取、映射和落盘。
    """脚本主入口，读取 UTIL 序列并写出准备结果。"""
    parser = argparse.ArgumentParser(description="Prepare UTIL raw data into a mapped bundle")  # 创建参数解析器。
    parser.add_argument("--config", default=str(ROOT / "configs" / "datasets" / "util.yaml"))  # UTIL 配置文件路径，默认指向仓库内配置。
    parser.add_argument("--raw-root", default=None)  # 原始数据目录覆盖值。
    parser.add_argument("--seq-ids", default=None)  # 要处理的序列列表。
    parser.add_argument("--output-root", default=None)  # 输出目录覆盖值。
    args = parser.parse_args(argv)  # 解析命令行参数。

    from liquidloc.common.tee_logger import print_args, print_dict
    print_args(args, "17_prepare_util_data")

    print("[17_util_data] 开始 | config=" + str(args.config), flush=True)
    try:
        config_path = Path(args.config)  # 把配置路径转成 Path，后面统一按路径对象处理。
        util_cfg = load_yaml_config(config_path)  # 读取 UTIL 配置，拿到 raw_root 和 field_mapping。
        print_dict(util_cfg, "UTIL 数据集配置")
        raw_root = _resolve_raw_root(config_path, args.raw_root)  # 解析原始数据目录。
        seq_ids = _resolve_seq_ids(raw_root, args.seq_ids)  # 解析这次要处理的序列列表。
        field_mapping = util_cfg.get("field_mapping") or {}  # 读取字段映射，没配置时就用空映射。
        print("[17_util_data] 配置已加载 | seq_count=" + str(len(seq_ids)) + " raw_root=" + str(raw_root), flush=True)

        if args.output_root is not None:  # 如果命令行显式指定输出目录。
            if not args.output_root.strip():  # 空字符串不允许。
                raise ValueError("UTIL --output-root must be a non-empty path")  # 这里直接报错，不默认猜路径。
            output_root = Path(args.output_root).resolve()  # 把输出目录规范成绝对路径。
        else:  # 否则用默认输出目录。
            output_root = (ROOT / "outputs" / "data_prep" / "util").resolve()  # 默认落在 util 准备目录。
        output_root.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
        print_dict({"config_path": str(config_path), "raw_root": str(raw_root), "seq_ids": seq_ids, "field_mapping": field_mapping, "output_root": str(output_root)}, "派生路径与参数")

        # 合同校验由 prepare_pipeline.run() 自动执行（run_dataset_checks + UTIL_REQUIRED_RAW_KEYS
        # + _reject_unsupported_util_tof_bridge），脚本层不再重复实现 read/mapping/check 逻辑，
        # 保证与 03 系列脚本走同一统一合同格式与输出文件名 {seq_id}_events.json（用户审查标准 #6/#7/#8）。
        from liquidloc.pipelines.prepare_pipeline import run as prepare_pipeline_run
        pipeline_cfg = {  # 组装 prepare pipeline 输入配置。
            "dataset_name": "util",  # 数据集名固定是 util。
            "raw_root": str(raw_root),  # 原始数据目录。
            "seq_ids": seq_ids,  # 要处理的序列列表。
            "field_mapping": field_mapping,  # 字段映射。
            "output_root": str(output_root),  # 输出目录。
        }  # pipeline 配置结束。
        print("[17_util_data] 正在运行 prepare_pipeline | seq_count=" + str(len(seq_ids)), flush=True)
        result = prepare_pipeline_run(pipeline_cfg)  # 走统一准备流水线，输出 {seq_id}_events.json + prepare_manifest.json。
        artifacts = list(result.artifacts)  # 收集产物路径。
        # 写一份 prepare_summary.json, 把 raw_root / output_root / seq_ids / artifact_count 留在
        # 顶层供下游脚本与测试消费 (prepare_manifest.json 不暴露 raw_root 顶层键).
        import json as _json
        summary_payload = {
            "raw_root": str(raw_root),
            "output_root": str(output_root),
            "seq_ids": list(seq_ids),
            "artifact_count": len(artifacts),
            "artifacts": [str(p) for p in artifacts],
        }
        (output_root / "prepare_summary.json").write_text(
            _json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        print("[17_util_data] 完成 | 返回码=<fail:prepare_pipeline>", flush=True)
        return _fail("prepare_pipeline", str(exc), error_type=type(exc).__name__)
    print("[17_util_data] 完成 | 返回码=0", flush=True)
    return _print_summary({"artifact_count": len(artifacts), "artifacts": artifacts, "exit_code": 0})  # 打印一个简短摘要。


if __name__ == "__main__":  # 只有直接运行这个脚本时才会进入这里。
    raise SystemExit(main())  # 用 main 的返回码结束进程，方便 shell 判断成功或失败。
