"""把 NTU VIRAL 原始数据准备成统一事件流。

这个脚本读取 `configs/datasets/ntu_viral.yaml`，把原始目录、字段映射和序列
列表整理成 `prepare_pipeline.run` 需要的配置字典。它不做领域清洗，只负责把
NTU VIRAL 原始数据接入统一准备流水线。
"""

from __future__ import annotations  # 允许后面的类型注解直接写成 `Path | None` 这种形式。

import argparse  # 解析命令行参数。
import json  # 打印 JSON 报告。
import sys  # 调整导入路径。
from pathlib import Path  # 统一处理路径。
from typing import Any  # 标注嵌套字典类型。

ROOT = Path(__file__).resolve().parents[1]  # 仓库根目录，后面所有默认路径都以这里为基准。
SRC = ROOT / "src"  # 源码目录，脚本直接运行时需要把它加入导入路径。
if str(SRC) not in sys.path:  # 如果源码目录还没进导入路径，就补进去。
    sys.path.insert(0, str(SRC))  # 放到最前面，避免导入到别处同名模块。

from liquidloc.common.config_utils import load_yaml_config  # 读取 YAML 配置。
from liquidloc.common.io_utils import dumps_json_text  # 严格 JSON 序列化，拒绝 NaN/Infinity。
from liquidloc.pipelines.prepare_pipeline import run  # 执行 prepare 流水线。


def _resolve_non_empty_path(
    raw_value: str | None,
    flag_name: str,
    default: Path | None = None,
    *,
    anchor: Path | None = None,
) -> Path:  # 规范命令行路径参数。
    """把命令行路径参数规范成非空绝对路径。"""
    if raw_value is None:  # 没传值时走默认值分支。
        if default is None:  # 默认值也没有就直接报错。
            raise ValueError(f"{flag_name} must be a non-empty path")  # 明确告诉调用者这个参数不能空着。
        resolved_default = default  # 先保留默认路径对象。
        if anchor is not None and not resolved_default.is_absolute():  # 默认值是相对路径时按锚点解释。
            resolved_default = anchor / resolved_default  # 把默认值补成锚点下路径。
        return resolved_default.resolve()  # 默认值也统一转成绝对路径。
    value = raw_value.strip()  # 先去掉首尾空白，避免空格伪装成有效路径。
    if not value:  # 真正空字符串也视为非法。
        raise ValueError(f"{flag_name} must be a non-empty path")  # 这里直接报错，不继续往下走。
    path = Path(value)  # 先转成路径对象。
    if anchor is not None and not path.is_absolute():  # 显式相对路径应按 project_root 解释，而不是当前工作目录。
        path = anchor / path  # 把相对路径补成锚点下路径。
    return path.resolve()  # 统一转成绝对路径，后面拼接更稳定。


def _resolve_cfg_path(project_root: Path, value: str) -> Path:  # 把配置里的 raw_root 解释成绝对路径。
    """把配置里的相对路径转成项目根目录下的绝对路径。"""
    if not isinstance(value, str) or not value.strip():  # 配置值必须是非空字符串。
        raise ValueError("ntu_viral config raw_root must be a non-empty path")  # 这里说明配置缺了关键字段。
    path = Path(value.strip())  # 先按路径对象处理。
    return path if path.is_absolute() else (project_root / path).resolve()  # 相对路径按项目根目录解释。


def _resolve_default_raw_root(project_root: Path, dataset_cfg: dict[str, Any]) -> Path:  # 选择 NTU VIRAL 的默认原始目录。
    """选择 NTU VIRAL 的默认原始目录。"""
    config_raw_root = _resolve_cfg_path(project_root, dataset_cfg["raw_root"])  # 先看配置指向哪里。
    return config_raw_root.resolve()  # 这里不额外回退，直接按配置解析。


def _ensure_existing_directory(path: Path, flag_name: str) -> Path:  # 确认路径存在且确实是目录。
    """确认路径存在且确实是目录。"""
    if not path.exists() or not path.is_dir():  # 不存在或者不是目录都不行。
        raise ValueError(f"{flag_name} must point to an existing directory")  # 这里直接报错，避免后面继续跑空路径。
    return path  # 通过检查后原样返回。


def _list_sequence_ids(raw_root: Path) -> list[str]:  # 列出原始目录下的序列 ID。
    """列出原始目录下的序列 ID。"""
    return [path.name for path in sorted(raw_root.iterdir()) if path.is_dir() and path.name != "config" and not path.name.startswith(".")]  # 跳过配置和隐藏目录。


def _build_pipeline_cfg(project_root: Path, raw_root: Path | None, output_root: Path | None) -> dict[str, Any]:  # 组装 prepare pipeline 需要的配置。
    """组装 prepare pipeline 需要的配置。"""
    dataset_cfg = load_yaml_config(project_root / "configs" / "datasets" / "ntu_viral.yaml")  # 读取 NTU VIRAL 配置。
    if not isinstance(dataset_cfg.get("raw_root"), str) or not dataset_cfg["raw_root"].strip():  # raw_root 必须有效。
        raise ValueError("ntu_viral config raw_root must be a non-empty path")  # 配置里缺 raw_root 就不能继续。
    field_mapping = dataset_cfg.get("field_mapping")  # 先拿字段映射。
    if not isinstance(field_mapping, dict) or not field_mapping:  # 映射必须存在且非空。
        raise ValueError("field_mapping must be a non-empty mapping")  # 否则准备阶段没法对字段做转换。
    resolved_raw_root = _ensure_existing_directory(raw_root or _resolve_default_raw_root(project_root, dataset_cfg), "--raw-root")  # 先决定原始目录。
    seq_ids = _list_sequence_ids(resolved_raw_root)  # 再列出序列 ID。
    if not seq_ids:  # 没有序列就不能继续。
        raise ValueError("ntu_viral seq_ids must be a non-empty list")  # 这里说明原始目录里没有可用数据。
    cfg: dict[str, Any] = {  # 下面组装的是 prepare pipeline 的输入配置。
        "dataset_name": dataset_cfg["dataset_name"],  # 数据集名原样传给流水线。
        "raw_root": str(resolved_raw_root),  # 这里写入最终 raw_root。
        "seq_ids": seq_ids,  # 这里写入全部序列 ID。
        "field_mapping": field_mapping,  # 这里写入字段映射。
        "output_root": str(output_root or (project_root / "outputs" / "prepare_ntu_viral")),  # 这里写入默认输出目录。
    }  # prepare 配置到这里结束。
    from liquidloc.common.tee_logger import print_dict
    print_dict(cfg, "流水线配置 (pipeline_cfg)")
    return cfg


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


def main(argv: list[str] | None = None) -> int:  # 脚本主入口，运行准备流水线并打印结果。
    """脚本主入口，运行准备流水线并打印结果。"""
    print("[03_ntu_viral] 开始 | 准备 NTU VIRAL 数据", flush=True)
    parser = argparse.ArgumentParser(description="Prepare NTU VIRAL raw data into unified events.")  # 定义参数。
    parser.add_argument("--project-root", default=str(ROOT))  # 默认项目根目录就是仓库根。
    parser.add_argument("--raw-root", default=None)  # 允许显式覆盖原始数据根目录。
    parser.add_argument("--output-root", default=None)  # 允许显式覆盖输出目录。
    args = parser.parse_args(argv)  # 解析命令行。
    from liquidloc.common.tee_logger import print_args, print_dict
    print_args(args, "03_prepare_ntu_viral_data")
    print("[03_ntu_viral] 解析路径", flush=True)
    try:
        project_root = _resolve_non_empty_path(args.project_root, "--project-root", ROOT)  # 规范项目根目录。
        raw_root = _resolve_non_empty_path(args.raw_root, "--raw-root", anchor=project_root) if args.raw_root is not None else None  # 显式相对路径按 project_root 解释。
        output_root = _resolve_non_empty_path(args.output_root, "--output-root", anchor=project_root) if args.output_root is not None else None  # 显式相对路径按 project_root 解释。
        print_dict(
            {
                "project_root": str(project_root),
                "raw_root": str(raw_root) if raw_root else None,
                "output_root": str(output_root) if output_root else None,
            },
            "路径配置",
        )
        print("[03_ntu_viral] 运行准备流水线", flush=True)
        # 合同校验由 prepare_pipeline.run() 自动执行（run_dataset_checks + REQUIRED_RAW_KEYS），
        # 字段漂移立即 abort；脚本层不需要重复调用 inspect_* 校验函数。
        result = run(_build_pipeline_cfg(project_root, raw_root, output_root))  # 调用准备流水线。
    except Exception as exc:
        _rc = _fail("prepare_pipeline", str(exc), error_type=type(exc).__name__)
        print(f"[03_ntu_viral] 完成 | 返回码={_rc}", flush=True)
        return _rc
    payload = {"stage_name": result.stage_name, "artifacts": result.artifacts, "metadata": result.metadata, "exit_code": 0}  # 只保留关键信息。
    _rc = _print_summary(payload)  # 打印给调用方。
    print(f"[03_ntu_viral] 完成 | 返回码={_rc}", flush=True)
    return _rc


if __name__ == "__main__":  # 直接执行时走主入口。
    raise SystemExit(main())  # 用返回码结束进程。
