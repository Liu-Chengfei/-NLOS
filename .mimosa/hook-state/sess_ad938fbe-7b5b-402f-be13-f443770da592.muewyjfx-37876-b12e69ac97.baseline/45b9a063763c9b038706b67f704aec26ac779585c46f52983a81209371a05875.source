"""项目路径工具。

职责：
    集中处理项目根目录、标准目录映射和输出路径拼接，避免上层到处
    手写 `Path(__file__)...` 这种容易漂移的路径推导。

上游依赖：
    - liquidloc.common.constants     — 复用 DEFAULT_OUTPUT_DIRS 构造输出子目录

下游调用者：
    - liquidloc.common.__init__      — 统一导出路径工具给上层
    - liquidloc.common.log_utils     — 用 build_output_path 构造日志文件路径
    - liquidloc.common.io_utils      — 间接通过 log_utils 使用
    - liquidloc.pipelines.*          — 流水线层用 get_standard_dirs / build_output_path 定位输出目录
    - scripts / notebooks            — 脚本用 get_project_root 定位仓库根

核心变量：
    - 无模块级变量，全部通过函数返回
"""

from __future__ import annotations  # 允许函数签名里引用未来定义的类型。

from pathlib import Path  # 用 Path 统一处理路径对象。
from typing import Any  # 用于类型注解。

from liquidloc.common.constants import DEFAULT_OUTPUT_DIRS  # 复用默认输出目录列表。
from liquidloc.common.validation import is_string_like  # 复用统一字符串类型校验。
from liquidloc.common.validation import validate_path_component  # 复用路径组件穿越校验。

__all__ = (
    "get_project_root",
    "get_standard_dirs",
    "build_output_path",
    "resolve_output_root",
)


def _resolve_project_root_arg(project_root: str | Path | None = None) -> Path | None:  # 把外部传入的根目录参数转成 Path。
    """把外部传入的项目根目录参数规范成 `Path`。

    Args:
        project_root (str | Path | None): 调用方传入的根目录参数。
            如果为 None，返回 None 交给上层兜底；
            如果为字符串，必须非空，否则抛出 ValueError；
            其他类型统一通过 Path().resolve() 转成绝对路径。

    Returns:
        Path | None: 解析后的绝对路径，或 None（当 project_root 为 None 时）。

    Raises:
        ValueError: 当 project_root 是空字符串或空白 Path 时抛出。
        TypeError: 当 project_root 不是 str/Path/None 类型时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"project_root": str(project_root) if project_root is not None else None}, "_resolve_project_root_arg 入口参数")
    if project_root is None:  # 如果调用方没有传，就让上层自己推导。
        return None  # 返回 None 交给 get_project_root 兜底。
    if not isinstance(project_root, (str, Path)):  # 前置类型检查，拒绝非法类型。
        raise TypeError(f"project_root must be str, Path or None, got {type(project_root).__name__}")
    if is_string_like(project_root):  # 字符串路径需要先做空白校验。
        if not str(project_root).strip():  # 空字符串没有路径意义。
            raise ValueError(f"project_root must be a non-empty path, got {project_root!r}")  # 空路径直接拒绝。
        return Path(project_root).expanduser().resolve()  # 展开 ~ 后转成绝对 Path。
    # Path 类型：先转 Path，再统一做空白校验（与 resolve_output_root 保持一致）。
    path_obj = Path(project_root)
    if not str(path_obj).strip():  # 空白 Path（如 Path("")、Path("  ")）没有路径意义。
        raise ValueError(f"project_root must be a non-empty path, got {project_root!r}")  # 空白路径直接拒绝。
    return path_obj.expanduser().resolve()  # 展开 ~ 后转成绝对 Path。


def get_project_root(project_root: str | Path | None = None) -> Path:  # 返回项目根目录。
    """返回项目根目录。

    如果调用方显式传入根目录，就优先使用；否则根据当前文件位置回推。

    Args:
        project_root (str | Path | None): 可选的项目根目录参数。
            传入时优先使用；不传时按当前文件位置回推仓库根目录
            （当前文件位于 src/liquidloc/common/，向上 3 层即为仓库根）。

    Returns:
        Path: 项目根目录的绝对路径。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"project_root": str(project_root) if project_root is not None else None}, "get_project_root 入口参数")
    resolved_root = _resolve_project_root_arg(project_root)  # 先看调用方有没有显式给根目录。
    if resolved_root is not None:  # 传了就直接用。
        return resolved_root  # 返回外部指定的根目录。
    from liquidloc.common.config_utils import find_project_root  # 延迟导入避免循环依赖。
    return find_project_root()  # 基于 marker 文件查找项目根，比 parents[N] 更健壮。


def get_standard_dirs(project_root: str | Path | None = None) -> dict[str, Any]:  # 构造仓库里常用的目录映射。
    """构造仓库里最常用的一组标准目录映射。

    Args:
        project_root (str | Path | None): 可选的项目根目录参数，
            传入时优先使用；不传时自动推导。

    Returns:
        dict: 标准目录映射字典，包含以下键：
            - project_root (Path): 项目根目录。
            - configs (Path): 配置目录。
            - data (Path): 数据目录。
            - outputs (Path): 输出根目录。
            - src (Path): 源码目录。
            - tests (Path): 测试目录。
            - output_subdirs (dict[str, Path]): 输出子目录映射，
              键名来自 DEFAULT_OUTPUT_DIRS。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"project_root": str(project_root) if project_root is not None else None}, "get_standard_dirs 入口参数")
    root = get_project_root(project_root)  # 先拿到项目根目录。
    outputs_dir = root / "outputs"  # outputs 是统一的运行产物根目录。
    dirs_map = {  # 先准备标准目录映射。
        "project_root": root,  # 项目根目录。
        "configs": root / "configs",  # 配置目录。
        "data": root / "data",  # 数据目录。
        "outputs": outputs_dir,  # 输出目录。
        "src": root / "src",  # 源码目录。
        "tests": root / "tests",  # 测试目录。
    }  # 基础映射结束。
    dirs_map["output_subdirs"] = {name: outputs_dir / name for name in DEFAULT_OUTPUT_DIRS}  # 再补上常用输出子目录。
    return dirs_map  # 返回目录映射。


def build_output_path(*parts: str, project_root: str | Path | None = None) -> Path:  # 把相对输出路径拼到 outputs 下。
    """把相对输出路径拼到标准 outputs 目录下。

    对每个 path part 调用 ``validate_path_component`` 校验，拒绝含
    ``..``、``/``、``\\``、null 字节的路径组件，防止路径穿越和绝对
    路径注入。

    Args:
        *parts (str): 要拼接的路径片段，例如 "metrics", "mini_metrics.csv"。
            每个片段必须是合法的相对路径组件，不允许含目录分隔符或穿越符。
        project_root (str | Path | None): 可选的项目根目录参数，
            传入时优先使用；不传时自动推导。

    Returns:
        Path: 拼接后的完整输出路径（outputs / *parts）。

    Raises:
        ValueError: 当 path part 包含 ``..``、``/``、``\\`` 或 null 字节时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"parts": parts, "project_root": str(project_root) if project_root is not None else None}, "build_output_path 入口参数")
    dirs = get_standard_dirs(project_root)  # 先拿到标准目录映射。
    for p in parts:  # 校验每个路径片段，防止穿越和绝对路径注入。
        validate_path_component(str(p), name="path part")
    return dirs["outputs"].joinpath(*parts)  # 在 outputs 下拼接子路径。


def resolve_output_root(cfg: dict[str, Any], default_name: str) -> Path:  # 统一的输出根目录解析函数。
    """从配置字典中解析 output_root，所有 pipeline 共用此函数。

    统一处理：None → 默认路径、空白字符串 → 报错、相对路径 → 按 project_root 解析。

    Args:
        cfg: 配置字典，可包含 output_root 和 project_root。
        default_name: 没有显式指定输出目录时使用的默认目录名。

    Returns:
        Path: 解析后的输出根目录绝对路径。

    Raises:
        ValueError: 当 output_root 为空白字符串、空白 Path 或 Path(".") 时。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"cfg": cfg, "default_name": default_name}, "resolve_output_root 入口参数")
    dirs = get_standard_dirs(cfg.get('project_root'))  # 先解析标准目录。
    raw_output_root = cfg.get('output_root')  # 读取显式输出根目录。
    if raw_output_root is None:  # 没指定时使用默认目录。
        return dirs['outputs'] / default_name
    # 前置类型检查：YAML 中 true/false/数字会被解析为非 str 类型，Path() 会静默转为字符串路径。
    if not isinstance(raw_output_root, (str, Path)):  # 拒绝非 str/Path 类型。
        raise TypeError(f"output_root must be str or Path, got {type(raw_output_root).__name__}: {raw_output_root!r}")
    # 统一转成 Path，再检查空白路径（Path("") / Path("  ") 在 str() 后为空或纯空白）。
    output_root = Path(raw_output_root)
    if not str(output_root).strip():  # 空白路径不允许，无论原始类型是 str 还是 Path。
        raise ValueError(f"output_root must not be blank, got {raw_output_root!r}")  # 空白路径会创建含空白的目录。
    # Path(".") 解析为当前工作目录，语义模糊且与 output_contract_schema 的校验标准不对齐。
    if str(output_root) == ".":  # 显式拒绝 Path(".") 输入。
        raise ValueError(f"output_root must not be '.', got {raw_output_root!r}")
    output_root = output_root.expanduser()  # 支持 ~ 展开。
    if output_root.is_absolute():  # 绝对路径直接用。
        return output_root.resolve()  # 返回绝对路径，消除 . / .. 等冗余。
    return (dirs['project_root'] / output_root).resolve()  # 相对路径按 project_root 解析，保证返回绝对路径。
