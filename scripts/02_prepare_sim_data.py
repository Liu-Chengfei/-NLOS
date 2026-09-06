"""把仿真原始数据准备成统一事件流。"""

from __future__ import annotations  # 允许使用现代类型注解写法。

import argparse  # 解析命令行参数。
import json  # 打印 JSON 报告。
import sys  # 调整导入路径。
from pathlib import Path  # 统一处理路径。
from typing import Any  # 标注嵌套字典类型。

ROOT = Path(__file__).resolve().parents[1]  # 仓库根目录。
SRC = ROOT / "src"  # 源码目录。
if str(SRC) not in sys.path:  # 把源码目录放进导入路径，方便脚本直接运行。
    sys.path.insert(0, str(SRC))  # 让脚本能直接导入项目模块。

from liquidloc.common.config_utils import load_yaml_config  # 读取 YAML 配置。
from liquidloc.common.io_utils import dumps_json_text  # 严格 JSON 序列化，拒绝 NaN/Infinity。
from liquidloc.dataio.manifests import inspect_sim_materialized_contract  # 检查 SIM raw 是否满足主表欠定几何合同（K1/K3 几何档位）。
from liquidloc.pipelines.prepare_pipeline import run  # 执行 prepare 流水线。


def _resolve_non_empty_path(
    raw_value: str | None,
    flag_name: str,
    default: Path | None = None,
    *,
    anchor: Path | None = None,
) -> Path:  # 把路径参数规范成非空绝对路径。
    """把命令行路径参数规范成非空绝对路径。

    参数
    ----------
    raw_value : str | None
        命令行传入的原始路径字符串；为 ``None`` 时使用默认值。
    flag_name : str
        命令行标志名称，用于错误消息定位。
    default : Path | None
        当 *raw_value* 为 ``None`` 时使用的默认路径；也为 ``None`` 时抛出异常。

    返回
    -------
    Path
        解析后的绝对路径。

    异常
    ------
    ValueError
        当 *raw_value* 去除首尾空白后为空字符串，或 *raw_value* 和 *default* 均为 ``None`` 时抛出。
    """
    if raw_value is None:  # 没传值时优先用默认值。
        if default is None:  # 默认值也没有就直接报错。
            raise ValueError(f"{flag_name} must be a non-empty path")  # 默认值也不能为空。
        resolved_default = default  # 先保留默认路径对象。
        if anchor is not None and not resolved_default.is_absolute():  # 默认值是相对路径时按锚点解释。
            resolved_default = anchor / resolved_default  # 把默认值补成锚点下的路径。
        return resolved_default.resolve()  # 默认值也统一转绝对路径。
    value = raw_value.strip()  # 先去掉首尾空白。
    if not value:  # 空字符串也算非法。
        raise ValueError(f"{flag_name} must be a non-empty path")  # 空白字符串也不接受。
    path = Path(value)  # 先转成路径对象。
    if anchor is not None and not path.is_absolute():  # 显式相对路径应按 project_root 解释，而不是当前工作目录。
        path = anchor / path  # 把相对路径补成锚点下路径。
    return path.resolve()  # 返回标准绝对路径。


def _resolve_cfg_path(project_root: Path, value: str) -> Path:  # 把配置里的 raw_root 解释成项目根目录下的路径。
    """把配置里的相对路径转成项目根目录下的绝对路径。

    参数
    ----------
    project_root : Path
        项目根目录，用于解释相对路径。
    value : str
        配置文件中的路径字符串，不允许为空或纯空白。

    返回
    -------
    Path
        解析后的路径；绝对路径原样返回，相对路径按 *project_root* 解释。

    异常
    ------
    ValueError
        当 *value* 去除首尾空白后为空字符串时抛出。
    """
    if not value.strip():  # 配置值不能为空。
        raise ValueError("sim config raw_root must be a non-empty path")  # sim 配置必须给出有效 raw_root。
    path = Path(value)  # 先按路径对象处理。
    return path if path.is_absolute() else project_root / path  # 相对路径按项目根目录解释。


def _resolve_default_raw_root(project_root: Path, dataset_cfg: dict[str, Any]) -> Path:  # 选择默认仿真原始目录。
    """选择仿真数据默认原始目录。

    优先使用配置文件中 ``raw_root`` 指向的目录（要求该目录存在且
    含至少一个子目录），否则直接失败，避免把测试夹具静默带入实验链。

    参数
    ----------
    project_root : Path
        项目根目录。
    dataset_cfg : dict[str, Any]
        仿真数据集配置字典，必须包含 ``raw_root`` 键。

    返回
    -------
    Path
        最终选定的仿真原始数据根目录。
    """
    config_raw_root = _resolve_cfg_path(project_root, dataset_cfg["raw_root"])  # 先看配置指向哪里。
    if config_raw_root.is_dir() and any(path.is_dir() for path in config_raw_root.iterdir()):  # 配置目录有效就直接用。
        return config_raw_root  # 配置目录可用时优先使用它。
    raise FileNotFoundError(
        "sim raw_root is missing or empty; expected materialized sim raw sequences at "
        f"{config_raw_root}. Run scripts/02_generate_sim_raw.py first or point --raw-root to a populated sim raw directory."
    )


def _expand_seq_ids_with_seeds(
    seq_ids: list[str],
    n_seed: int,
) -> list[str]:
    """将序列 ID 列表扩展为多种子变体。

    当 n_seed > 1 时，每个原始 seq_id 生成 n_seed 个带种子后缀的变体
    （例如 sim_curve_01 → sim_curve_01_seed0 … sim_curve_01_seed29）。
    n_seed=1 时返回原列表不变。
    """
    if n_seed <= 1:
        return list(seq_ids)
    expanded: list[str] = []
    for seq_id in seq_ids:
        for seed_idx in range(n_seed):
            expanded.append(f"{seq_id}_seed{seed_idx}")
    return expanded


def _ensure_seed_variant_dirs(
    resolved_raw_root: Path,
    base_seq_ids: list[str],
    n_seed: int,
) -> None:
    """为多种子变体创建包含正确元数据的目录。

    当 n_seed > 1 时，每个基础序列生成 n_seed 个种子变体目录
    （如 sim_curve_01_seed0 … sim_curve_01_seed29）。
    每个变体目录包含：
    - 指向基础序列数据的符号链接（节省磁盘空间）
    - 独立的 sim_meta.json（含 n_seed 和 seed_id）
    - 独立的 anchor_layout.json（layout_id 匹配变体名称）
    这确保 inspect_sim_materialized_contract 合同检查通过，
    且 prepare 管道能正确读取每个种子变体的原始数据。

    在 Windows 上，符号链接可能需要管理员权限或开发者模式；
    此时回退使用 junction（mklink /J）。
    """
    import os  # 平台相关的符号链接创建。

    for seq_id in base_seq_ids:
        base_dir = resolved_raw_root / seq_id
        if not base_dir.is_dir():
            continue  # 跳过非目录条目。
        base_sim_meta_path = base_dir / "sim_meta.json"
        base_anchor_layout_path = base_dir / "anchor_layout.json"
        base_sim_meta = json.loads(base_sim_meta_path.read_text(encoding="utf-8")) if base_sim_meta_path.is_file() else {}
        base_anchor_layout = json.loads(base_anchor_layout_path.read_text(encoding="utf-8")) if base_anchor_layout_path.is_file() else {}
        for seed_idx in range(n_seed):
            variant_name = f"{seq_id}_seed{seed_idx}"
            variant_dir = resolved_raw_root / variant_name
            if variant_dir.exists():
                continue  # 已存在则跳过。
            variant_dir.mkdir(parents=True, exist_ok=True)
            # 为每个数据文件创建指向基础序列的符号链接。
            for fname in ["imu.json", "uwb.json", "vio.json", "gt.json"]:
                src = base_dir / fname
                dst = variant_dir / fname
                if src.is_file() and not dst.exists():
                    try:
                        os.symlink(src, dst, target_is_directory=True)
                    except (OSError, NotImplementedError):
                        # Windows symlink 需要 SeCreateSymbolicLinkPrivilege，回退使用 junction。
                        try:
                            import subprocess
                            # mklink /J <junction_path> <target_path>：junction 不需要权限，参数顺序是 dst src。
                            subprocess.run(
                                ["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                                check=True,
                                capture_output=True,
                                text=True,
                            )
                        except Exception:
                            # 最后的回退：复制目录（保留时间戳）。
                            import shutil
                            shutil.copytree(src, dst, dirs_exist_ok=True, copy_function=shutil.copy2)
            # 写入种子变体的 sim_meta.json。
            variant_sim_meta = dict(base_sim_meta)
            variant_sim_meta["seq_id"] = variant_name
            variant_sim_meta["seed_id"] = f"seed{seed_idx}"
            variant_sim_meta["n_seed"] = n_seed
            variant_sim_meta["base_seq_id"] = seq_id
            variant_sim_meta["is_seed_variant"] = True
            (variant_dir / "sim_meta.json").write_text(
                json.dumps(variant_sim_meta, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            # 写入种子变体的 anchor_layout.json（layout_id 匹配变体名称）。
            variant_anchor_layout = dict(base_anchor_layout)
            variant_anchor_layout["layout_id"] = variant_name
            variant_anchor_layout["base_layout_id"] = base_anchor_layout.get("base_layout_id", seq_id)
            (variant_dir / "anchor_layout.json").write_text(
                json.dumps(variant_anchor_layout, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )


def _build_scene_id_by_seq(
    seq_ids: list[str],
    resolved_raw_root: Path,
) -> dict[str, str]:
    """从 sim_meta.json 的 axes_override 生成 scene_id_by_seq。

    缺失时回退主表欠定默认 S(A0,N0,V0,K0,M0)。
    """
    scene_id_by_seq: dict[str, str] = {}
    for seq_id in seq_ids:
        sim_meta_path = resolved_raw_root / seq_id / "sim_meta.json"
        if sim_meta_path.is_file():
            try:
                sim_meta = json.loads(sim_meta_path.read_text(encoding="utf-8"))
                axes_override = sim_meta.get("axes_override") or {}
                if axes_override:
                    # P2 修复 (2026-09-02): 2026-08-31 协议重构后 G 已合并到 K，
                    # axis_order 不再含 G，scene_id 形如 S(A,N,V,K,M)。
                    axis_order = ["A", "N", "V", "K", "M"]
                    parts = [axes_override[ax] for ax in axis_order if ax in axes_override]
                    if parts:
                        scene_id_by_seq[seq_id] = f"S({','.join(parts)})"
                        continue
            except Exception:
                pass  # 解析失败时回退到主表欠定默认
        scene_id_by_seq[seq_id] = "S(A0,N0,V0,K3,M0)"  # 五轴档位协议：主表欠定默认 K3（K 轴仅 K0/K1/K3，锚数全档固定 4）+ M0（无模态缺失）
    return scene_id_by_seq


def _build_pipeline_cfg(project_root: Path, raw_root: Path | None, output_root: Path | None, n_seed: int = 1) -> dict[str, Any]:  # 组装 prepare pipeline 输入配置。
    """组装 prepare pipeline 需要的配置。

    参数
    ----------
    project_root : Path
        项目根目录，用于读取数据集配置。
    raw_root : Path | None
        命令行指定的原始数据根目录；为 ``None`` 时从配置推断。
    output_root : Path | None
        命令行指定的输出目录；为 ``None`` 时使用默认路径。
    n_seed : int
        种子变体数量。当 n_seed > 1 时，每个基础序列扩展为 n_seed 个
        带种子后缀的 seq_id（例如 sim_curve_01_seed0 … sim_curve_01_seed29）。

    返回
    -------
    dict[str, Any]
        prepare pipeline 所需的完整配置字典，包含 ``dataset_name``、
        ``raw_root``、``seq_ids``、``output_root`` 和 ``scene_id_by_seq``。

    异常
    ------
    ValueError
        当数据集配置中 ``raw_root`` 缺失或为空，或序列目录为空时抛出。
    """
    dataset_cfg = load_yaml_config(project_root / "configs" / "datasets" / "sim.yaml")  # 读取仿真数据集配置。
    if not isinstance(dataset_cfg.get("raw_root"), str) or not dataset_cfg["raw_root"].strip():  # raw_root 必须有效。
        raise ValueError("sim config raw_root must be a non-empty path")  # sim 配置必须有可用 raw_root。
    resolved_raw_root = raw_root or _resolve_default_raw_root(project_root, dataset_cfg)  # 优先用命令行，否则用默认 raw_root。
    base_seq_ids = [p.name for p in sorted(resolved_raw_root.iterdir()) if p.is_dir()]  # 序列 ID 直接取子目录名。
    if not base_seq_ids:  # 没有序列就不能继续。
        raise ValueError("sim seq_ids must be a non-empty list")  # 没有序列就不能进入 prepare。
    seq_ids = _expand_seq_ids_with_seeds(base_seq_ids, n_seed)  # 多种子时扩展 seq_ids。
    # 多种子时，为每个种子变体创建包含正确元数据的目录，
    # 使 read_sim_sequence 能从 raw_root / seq_id 正确读取原始数据。
    if n_seed > 1:
        _ensure_seed_variant_dirs(resolved_raw_root, base_seq_ids, n_seed)
    # BUG-003 修复 (2026-09-06 §10 审计): sim_e9_main 是 seed0..seed9 嵌套结构，
    # 每个 seedN/ 才是 sim 序列容器（如 seed0/sim_curve_01/）。直接对顶层
    # sim_e9_main/ 做 contract 检查会把 seed 目录当序列，导致 10/10 missing anchor_layout。
    # n_seed > 1 时对每个 seed 目录分别做 contract 检查；n_seed==1 时直接对目录本身检查。
    if n_seed > 1:
        seed_dirs = [resolved_raw_root / s for s in base_seq_ids if (resolved_raw_root / s).is_dir()]
        if not seed_dirs:
            raise RuntimeError(
                f"n_seed={n_seed} but no seed subdirs found under {resolved_raw_root}"
            )
        for seed_dir in seed_dirs:
            sim_contract_report = inspect_sim_materialized_contract(seed_dir)
            if not sim_contract_report["is_valid"]:
                raise RuntimeError(
                    f"sim seed {seed_dir.name} does not satisfy the paper-grade materialized SIM contract "
                    f"(allowed K∈{{K0,K1,K3}}); report={sim_contract_report}. "
                    f"Re-run scripts/02_generate_sim_raw.py or point --raw-root to a protocol-trajectory "
                    f"undetermined-geometry sim raw directory."
                )
    else:
        sim_contract_report = inspect_sim_materialized_contract(resolved_raw_root)
        if not sim_contract_report["is_valid"]:
            raise RuntimeError(
                "sim raw_root does not satisfy the paper-grade materialized SIM contract "
                "(allowed K∈{K0,K1,K3}); "
                f"report={sim_contract_report}. Re-run scripts/02_generate_sim_raw.py or point "
                "--raw-root to a protocol-trajectory undetermined-geometry sim raw directory."
            )
    scene_id_by_seq = _build_scene_id_by_seq(seq_ids, resolved_raw_root)  # 从 sim_meta.json 读取 axes_override。
    cfg: dict[str, Any] = {  # 下面组装的是 prepare pipeline 的输入配置。
        "dataset_name": dataset_cfg["dataset_name"],  # 这里保留数据集名给流水线使用。
        "raw_root": str(resolved_raw_root),  # 这里写入最终 raw_root。
        "seq_ids": seq_ids,  # 这里写入全部序列 ID（多种子时含 seed 后缀）。
        # BUG-010 修复 (2026-09-06 §10.2 审计): 之前默认 outputs/prepare_sim 不含数据集名，
        # 导致 sim_e9_main 数据准备好后落到 outputs/prepare_sim 而非 configs/datasets/sim.yaml 的 prepare_root。
        # 现在跟 sim.yaml.prepare_root 对齐（data/raw/sim_e9_main → outputs/prepare_sim_e9_main）。
        "output_root": str(output_root or (project_root / dataset_cfg.get("prepare_root", "outputs/prepare_sim"))),  # 这里写入默认输出目录。
        "scene_id_by_seq": scene_id_by_seq,  # v2: 从 sim_meta.json 读取, 不再硬编码 baseline.
        "n_seed": n_seed,  # 种子变体数量，供下游消费方参考。
    }  # pipeline 配置到这里结束。
    from liquidloc.common.tee_logger import print_dict
    print_dict(cfg, "流水线配置 (pipeline_cfg)")
    return cfg  # 返回给真正的流水线执行。


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


def main(argv: list[str] | None = None) -> int:  # 脚本主入口，执行 prepare 并打印结果。
    """脚本主入口，运行准备流水线并打印结果。

    参数
    ----------
    argv : list[str] | None
        命令行参数列表；为 ``None`` 时从 ``sys.argv`` 读取。

    返回
    -------
    int
        退出码，0 表示正常结束。

    注意
    -----
    本函数会把 prepare 流水线结果以 JSON 格式输出到标准输出。
    如果路径参数为空或序列目录为空，将抛出 ``ValueError``。
    """
    print("[02_sim_prepare] 开始 | 准备仿真数据", flush=True)
    parser = argparse.ArgumentParser(description="Prepare simulation raw data into unified events.")  # 定义参数。
    parser.add_argument("--project-root", default=str(ROOT))  # 默认项目根目录就是仓库根。
    parser.add_argument("--raw-root", default=None)  # 允许显式覆盖原始数据根目录。
    parser.add_argument("--output-root", default=None)  # 允许显式覆盖输出目录。
    parser.add_argument("--n-seed", type=int, default=1, help="种子变体数量（默认1，即不扩展）。当 n_seed>1 时，每个基础序列生成 n_seed 个带 seed 后缀的 seq_id。")  # 多种子支持。
    args = parser.parse_args(argv)  # 解析命令行。
    from liquidloc.common.tee_logger import print_args, print_dict
    print_args(args, "02_prepare_sim_data")
    print("[02_sim_prepare] 解析路径", flush=True)
    try:
        project_root = _resolve_non_empty_path(args.project_root, "--project-root", ROOT)  # 规范项目根目录。
        raw_root = _resolve_non_empty_path(args.raw_root, "--raw-root", anchor=project_root) if args.raw_root is not None else None  # 显式相对路径按 project_root 解释。
        output_root = _resolve_non_empty_path(args.output_root, "--output-root", anchor=project_root) if args.output_root is not None else None  # 显式相对路径按 project_root 解释。
        print_dict(
            {
                "project_root": str(project_root),
                "raw_root": str(raw_root) if raw_root else None,
                "output_root": str(output_root) if output_root else None,
                "n_seed": args.n_seed,
            },
            "路径配置",
        )
        print("[02_sim_prepare] 运行准备流水线", flush=True)
        result = run(_build_pipeline_cfg(project_root, raw_root, output_root, n_seed=args.n_seed))  # 调用准备流水线。
    except Exception as exc:
        _rc = _fail("prepare_pipeline", str(exc), error_type=type(exc).__name__)
        print(f"[02_sim_prepare] 完成 | 返回码={_rc}", flush=True)
        return _rc
    payload = {"stage_name": result.stage_name, "artifacts": result.artifacts, "metadata": result.metadata, "exit_code": 0}  # 只保留关键信息。
    _rc = _print_summary(payload)  # 打印给调用方。
    print(f"[02_sim_prepare] 完成 | 返回码={_rc}", flush=True)
    return _rc


if __name__ == "__main__":  # 直接执行时走主入口。
    raise SystemExit(main())