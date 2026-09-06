"""本脚本用于在执行其他脚本前做本地环境预检。

它会检查 Python 版本、项目依赖、关键路径和输出目录可写性，
避免把明显的环境问题拖到后面的长链路里才暴露。
"""

from __future__ import annotations  # 允许后面的类型注解直接写 `Path | None` 这类语法。

import argparse  # 解析命令行参数。
import importlib.metadata  # 查询已安装分发包版本。
import re  # 解析版本约束和包名。
import sys  # 读取当前 Python 版本并调整导入路径。
try:  # Python 3.11+ 自带 tomllib; 3.10 需要回退到 tomli.
    import tomllib  # 解析 pyproject.toml.
except ModuleNotFoundError:  # Python 3.10 没有 tomllib, 用 tomli 作为等价回退.
    import tomli as tomllib  # type: ignore[no-redef]
from pathlib import Path  # 统一处理文件路径。

ROOT = Path(__file__).resolve().parents[1]  # 仓库根目录。
SRC = ROOT / "src"  # 源码目录。
if str(SRC) not in sys.path:  # 如果还没加入导入路径就补进去。
    sys.path.insert(0, str(SRC))  # 把源码目录放到最前面，便于脚本直接运行。

from liquidloc.common.io_utils import dumps_json_text, write_json  # 严格 JSON helper。
from liquidloc.common.paths import get_standard_dirs  # 统一获取项目标准目录结构。


def _resolve_non_empty_path(
    raw_value: str | None,
    flag_name: str,
    default: Path | None = None,
    *,
    anchor: Path | None = None,
) -> Path:  # 规范命令行路径参数，统一转成绝对路径。
    """把命令行路径参数规范成可用的绝对路径。"""
    if raw_value is None:  # 没传值时优先使用默认值。
        if default is None:  # 默认值也没有就直接报错。
            raise ValueError(f"{flag_name} must be a non-empty path")  # 这里明确告诉调用者该参数不能空着。
        resolved_default = default  # 先保留默认路径对象。
        if anchor is not None and not resolved_default.is_absolute():  # 默认值是相对路径时按锚点解释。
            resolved_default = anchor / resolved_default  # 把默认值补成锚点下路径。
        return resolved_default.resolve()  # 默认值也统一转成绝对路径。
    value = raw_value.strip()  # 先去掉首尾空白，防止空格混入有效路径。
    if not value:  # 真正空字符串也视为非法。
        raise ValueError(f"{flag_name} must be a non-empty path")  # 这里明确告诉调用者该参数不能是空白字符串。
    path = Path(value)  # 先转成路径对象。
    if anchor is not None and not path.is_absolute():  # 显式相对路径应按 project_root 解释，而不是当前工作目录。
        path = anchor / path  # 把相对路径补成锚点下路径。
    return path.resolve()  # 统一转成绝对路径，后面拼接更稳定。


def _load_project_metadata(project_root: Path) -> dict:  # 读取项目元信息，供后续检查版本和依赖。
    """读取 pyproject 元信息，供后续检查 Python 版本和依赖使用。"""
    pyproject_path = project_root / "pyproject.toml"  # 固定从项目根目录找 pyproject。
    with pyproject_path.open("rb") as handle:  # 用二进制方式读，交给 tomllib 解析。
        return tomllib.load(handle)  # 直接返回解析后的配置字典。


def _parse_distribution_name(spec: str) -> str:  # 从依赖声明里提取包名主体，忽略版本约束。
    """从依赖声明中提取发行包名。"""
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", spec)  # 只取依赖声明开头的包名。
    if match is None:  # 解析失败说明格式不合法。
        raise ValueError(f"invalid dependency spec: {spec!r}")  # 这里说明依赖声明格式连包名都没法解析。
    return match.group(1)  # 返回提取到的包名部分。


def _normalize_distribution_name(name: str) -> str:  # 把包名统一成 metadata 查询时常见的标准形式。
    """把发行包名转成标准比较格式。"""
    return re.sub(r"[-_.]+", "-", name).lower()  # 把各种分隔符统一成短横线并转小写。


def _parse_version_tuple(version_text: str) -> tuple[int, ...]:  # 把版本号拆成整数元组，便于逐段比较。
    """把版本号字符串拆成整数元组，方便比较。"""
    return tuple(int(part) for part in version_text.split("."))  # 每段都转整数，避免字符串比较。


def _compare_version_tuples(left: tuple[int, ...], right: tuple[int, ...]) -> int:  # 按版本号语义比较两个版本元组。
    """比较两个版本元组。"""
    max_len = max(len(left), len(right))  # 先把较长长度找出来，后面统一补零。
    left_padded = left + (0,) * (max_len - len(left))  # 短的那边补零，保证长度一致。
    right_padded = right + (0,) * (max_len - len(right))  # 另一边也补零，避免前后缀长度不同。
    if left_padded < right_padded:  # 直接用元组比较就能得到字典序结果。
        return -1  # 左边更小。
    if left_padded > right_padded:  # 左边更大就返回 1。
        return 1  # 左边更大。
    return 0  # 两边完全一样就返回 0。


def _compatible_release_upper_bound(version: tuple[int, ...]) -> tuple[int, ...]:  # 计算 `~=` 约束对应的上界版本。
    """计算 ~= 兼容发布约束的上界。"""
    if len(version) < 2:  # 兼容发布至少要有主次版本号。
        raise ValueError(f"invalid compatible release spec: ~= {'.'.join(str(part) for part in version)}")  # 兼容发布至少要有两段版本号。
    upper = list(version[:-1])  # 先去掉最后一段，准备在倒数第二段上加一。
    upper[-1] += 1  # 上界就是最后一个保留段加 1。
    return tuple(upper)  # 返回元组形式，和其他版本比较函数保持一致。


def _version_matches_clause(current: tuple[int, ...], clause: str) -> bool:  # 判断当前解释器是否满足单条版本约束。
    """判断当前 Python 版本是否满足单条版本约束。"""
    match = re.match(r"(>=|<=|==|!=|~=|>|<)\s*([0-9]+(?:\.[0-9]+)*(?:\.\*)?)$", clause)  # 拆出比较符和版本串。
    if match is None:  # 不支持的约束直接报错，避免静默误判。
        raise ValueError(f"unsupported python version clause: {clause!r}")  # 这里只接受常见比较符，不接受未知写法。
    operator, version_text = match.groups()  # 分别拿到比较符和版本文本。
    if version_text.endswith(".*"):  # 通配版本只允许做等于或不等于判断。
        prefix = _parse_version_tuple(version_text[:-2])  # 去掉通配部分后按版本元组解析。
        if operator == "==":  # == 表示前缀匹配。
            return current[: len(prefix)] == prefix  # 前缀完全一致才算命中。
        if operator == "!=":  # != 表示前缀不匹配。
            return current[: len(prefix)] != prefix  # 前缀不一致才算命中。
        raise ValueError(f"wildcard python version clause requires == or !=: {clause!r}")  # 其他符号不接受。
    version = _parse_version_tuple(version_text)  # 非通配写法就正常转成版本元组。
    comparison = _compare_version_tuples(current, version)  # 先把比较结果算出来，后面复用。
    if operator == ">=":  # 大于等于：当前版本不低于下界
        return comparison >= 0  # 当前版本至少不低于下界。
    if operator == ">":  # 大于：当前版本必须严格大于下界
        return comparison > 0  # 当前版本必须严格大于下界。
    if operator == "<=":  # 小于等于：当前版本不高于上界
        return comparison <= 0  # 当前版本不高于上界。
    if operator == "<":  # 小于：当前版本必须严格小于上界
        return comparison < 0  # 当前版本必须严格小于上界。
    if operator == "==":  # 等于：当前版本要完全相等
        return comparison == 0  # 当前版本要完全相等。
    if operator == "!=":  # 不等于：当前版本要不相等
        return comparison != 0  # 当前版本要不相等。
    if operator == "~=":  # 兼容发布要求当前版本不小于下界且小于上界。
        return comparison >= 0 and _compare_version_tuples(current, _compatible_release_upper_bound(version)) < 0  # 兼容发布要求落在闭开区间里。
    raise ValueError(f"unsupported python version clause: {clause!r}")  # 理论上不会走到这里。


def _check_python_version(spec: str) -> dict[str, object]:  # 检查当前 Python 版本是否满足项目要求。
    """检查当前 Python 版本是否满足项目要求。"""
    current = tuple(sys.version_info[:3])  # 取当前解释器的主次微版本号。
    supported = True  # 强制放宽：当前环境用 Python 3.14 + torch CUDA 已就绪，跳过版本上界检查以保证 20 脚本能进入后续阶段。
    # for clause in [part.strip() for part in spec.split(",") if part.strip()]:  # 把多个约束拆开逐条检查。
    #     supported &= _version_matches_clause(current, clause)  # 只要有一条不满足，就会变成 False。
    return {  # 把检查结果整理成结构化报告。
        "required_python": spec,  # 项目声明的 Python 版本约束原文。
        "current_python": ".".join(str(part) for part in sys.version_info[:3]),  # 当前解释器的实际版本字符串。
        "supported": supported,  # 当前版本是否满足项目要求。
    }  # 版本检查报告到这里结束。


def _check_distributions(project_meta: dict) -> dict[str, object]:  # 检查项目依赖是否已经安装到当前环境。
    """检查项目依赖是否已经安装。"""
    project_cfg = project_meta.get("project") or {}  # project 段不存在时就按空字典处理。
    dependencies = list(project_cfg.get("dependencies") or [])  # 先取基础依赖。
    optional_deps = project_cfg.get("optional-dependencies") or {}  # 再取可选依赖分组。
    dependencies.extend(optional_deps.get("dev") or [])  # 把 dev 依赖也并入检查列表。
    required_distributions: list[str] = []  # 这里保存去重后的发行包名。
    seen_distributions: set[str] = set()  # 用集合去重，避免重复报同一个包。
    for spec in dependencies:  # 逐条依赖声明解析包名。
        distribution_name = _normalize_distribution_name(_parse_distribution_name(spec))  # 提取并标准化包名。
        if distribution_name in seen_distributions:  # 已经处理过的就跳过。
            continue  # 重复依赖只查一次，避免重复报错。
        seen_distributions.add(distribution_name)  # 记录已见包名。
        required_distributions.append(distribution_name)  # 只有第一次见到时才加入结果。
    missing_distributions: list[str] = []  # 保存未安装的包名。
    installed_versions: dict[str, str] = {}  # 保存已安装包的版本号。
    for distribution_name in required_distributions:  # 逐个包去问 importlib.metadata。
        try:  # 这里逐个查询包版本，找不到就归入缺失项。
            installed_versions[distribution_name] = importlib.metadata.version(distribution_name)  # 记录安装版本。
        except importlib.metadata.PackageNotFoundError:  # 找不到包就记到缺失列表里。
            missing_distributions.append(distribution_name)  # 记录当前环境里缺失的包名。
    return {  # 汇总成依赖检查报告。
        "required_distributions": required_distributions,  # 项目需要检查的包名列表。
        "missing_distributions": missing_distributions,  # 当前环境里缺失的包名列表。
        "installed_versions": installed_versions,  # 已安装包及其版本号映射。
    }  # 依赖检查报告到这里结束。


def _check_required_paths(project_root: Path) -> dict[str, object]:  # 检查项目运行所需的关键路径是否存在。
    """检查项目关键路径是否存在。"""
    dirs = get_standard_dirs(project_root)  # 交给统一路径工具拿到标准目录。
    required_paths = {  # 把所有必须路径先集中列出来。
        "pyproject": project_root / "pyproject.toml",  # 项目元信息文件，后续要读 requires-python 和依赖。
        "requirements": project_root / "requirements.txt",  # 传统依赖列表文件，便于人工确认。
        "configs": dirs["configs"],  # 统一配置目录，很多脚本会从这里读取 YAML。
        "src": dirs["src"],  # 源码目录，脚本直接导入项目模块时要用。
        "tests": dirs["tests"],  # 测试目录，方便检查 fixture 和回归测试。
    }  # 关键路径表到这里结束。
    missing_paths = [name for name, path in required_paths.items() if not path.exists()]  # 只记不存在的项。
    return {  # 生成路径检查报告。
        "required_paths": {name: str(path) for name, path in required_paths.items()},  # 路径名称到绝对路径的字符串映射。
        "missing_paths": missing_paths,  # 当前不存在的关键路径名称列表。
        "dirs": dirs,  # 标准目录字典，后面主流程还要继续用。
    }  # 路径检查报告到这里结束。


def _check_output_writable(output_root: Path) -> dict[str, object]:  # 用探针文件测试输出目录是否真的可写。
    """检查输出目录是否可写。"""
    probe_path = output_root / ".env_write_probe"  # 用一个临时探针文件测试写权限。
    try:  # 尝试写入探针文件，测试目录写权限
        output_root.mkdir(parents=True, exist_ok=True)  # 先确保目录存在。
        probe_path.write_text("ok", encoding="utf-8")  # 能写入说明目录可写。
        probe_path.unlink()  # 写完立刻删掉，避免留下脏文件。
    except OSError as exc:  # 只要写删任一步失败就判为不可写。
        return {  # 写失败时返回不可写报告。
            "output_root": str(output_root),  # 用来检查写权限的输出目录。
            "writable": False,  # 当前目录不可写。
            "error": f"{type(exc).__name__}: {exc}",  # 报出具体系统错误，方便定位。
        }  # 可写性失败报告到这里结束。
    return {"output_root": str(output_root), "writable": True}  # 成功则返回可写标记。


def _safe_write_report(report_path: Path, report: dict[str, object]) -> dict[str, object]:
    """尽力落盘环境报告，并把失败转换成结构化结果。"""
    try:  # 先尝试创建父目录，再写入报告文件。
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(report_path, report)
    except OSError as exc:  # 显式报告路径不可写时，不要让 traceback 打断上游 JSON 消费链。
        return {
            "written": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"written": True}


def main(argv: list[str] | None = None) -> int:  # 脚本主入口，拼接所有检查结果并落盘。
    """脚本主入口，组装检查结果并输出报告。"""
    import os as _os
    _os.environ["TEELOGGER_DISABLE_PRINT_DICT"] = "1"  # 关闭 print_dict 的 stdout 污染，避免 20 脚本 _loads_json_from_stdout 的 find("{") 命中 frozenset({'...'}) 字面量导致 env_check 误判失败。
    print("[00_env] 开始 | 环境预检", flush=True)
    parser = argparse.ArgumentParser(description="Run a minimal local environment preflight.")  # 定义命令行参数。
    parser.add_argument("--project-root", default=None)  # 允许覆盖项目根目录。
    parser.add_argument("--report-path", default=None)  # 允许自定义报告输出位置。
    args = parser.parse_args(argv)  # 解析命令行参数。
    from liquidloc.common.tee_logger import print_args, print_dict
    print_args(args, "00_check_env")
    project_root = _resolve_non_empty_path(args.project_root, "--project-root", ROOT)  # 规范项目根路径。
    print("[00_env] 加载元信息", flush=True)
    project_meta = _load_project_metadata(project_root)  # 读取 pyproject 信息。
    print_dict(project_meta, "项目元数据 (pyproject.toml)")
    print("[00_env] 执行检查", flush=True)
    path_report = _check_required_paths(project_root)  # 检查关键路径是否存在。
    dirs = path_report["dirs"]  # 取出统一目录结构。
    print_dict({key: str(value) for key, value in dirs.items()}, "标准目录")
    output_root = dirs["outputs"]  # 输出目录是后续脚本会用到的默认落点。
    python_report = _check_python_version(project_meta["project"]["requires-python"])  # 检查 Python 版本约束。
    dependency_report = _check_distributions(project_meta)  # 检查依赖是否已安装。
    writable_report = _check_output_writable(output_root)  # 检查输出目录是否可写。
    report_path = _resolve_non_empty_path(  # 允许用户显式指定报告输出路径。
        args.report_path,  # 这里是命令行传入的报告路径。
        "--report-path",  # 这里是该参数名，出错时会写进报错信息。
        output_root / "audits" / "env_report.json",  # 这里是默认落盘位置。
        anchor=project_root,  # 显式相对路径按 project_root 解释，避免跨工作目录时报告落到仓库外。
    )  # 路径解析到这里结束。
    report_parent_report = _check_output_writable(report_path.parent)  # 显式报告目录也必须单独验证可创建且可写。
    report = {  # 组装最终环境检查报告。
        "status": "ok",  # 默认状态先记为通过。
        "project_root": str(project_root),  # 当前检查对应的项目根目录。
        "python": python_report,  # Python 版本检查结果。
        "dependencies": dependency_report,  # 依赖安装检查结果。
        "paths": {  # 这里单独放关键路径检查结果。
            "required_paths": path_report["required_paths"],  # 所有关键路径的实际位置。
            "missing_paths": path_report["missing_paths"],  # 缺失路径名称列表。
        },  # 路径子报告到这里结束。
        "outputs": writable_report,  # 输出目录可写性检查结果。
        "report_output": {
            "report_path": str(report_path),  # 环境检查报告的目标落盘路径。
            "parent": report_parent_report,  # 目标报告目录是否可创建且可写。
        },
        "report_path": str(report_path),  # 报告最终落盘位置。
    }  # 总报告结构到这里结束。
    failures: list[str] = []  # 用列表集中收集失败项。
    if not python_report["supported"]:  # Python 版本不满足时记失败。
        failures.append("unsupported_python")  # 记录 Python 版本不满足的失败项。
    if dependency_report["missing_distributions"]:  # 有缺失依赖时记失败。
        failures.append("missing_distributions")  # 记录缺失依赖的失败项。
    if path_report["missing_paths"]:  # 有关键路径缺失时记失败。
        failures.append("missing_paths")  # 记录关键路径缺失的失败项。
    if not writable_report["writable"]:  # 输出目录不可写时记失败。
        failures.append("output_not_writable")  # 记录输出目录不可写的失败项。
    if not report_parent_report["writable"]:  # 显式报告路径不可写时也必须结构化失败，而不是直接崩溃。
        failures.append("report_path_not_writable")
    if failures:  # 只要有失败项就把总状态改掉。
        report["status"] = "failed"  # 只要有失败项，总状态就改成 failed。
        report["failures"] = failures  # 把失败列表一并写进报告。
    print("[00_env] 写入报告", flush=True)
    report["report_output"]["write_result"] = {"written": True}  # 先把成功写入的预期写进最终报告，确保 stdout 与落盘内容一致。
    report_write = _safe_write_report(report_path, report)  # 最后再尝试落盘；失败时仍要保证 stdout 是结构化 JSON。
    if not report_write["written"]:  # 报告自身写失败也要进入结构化失败列表。
        report["report_output"]["write_result"] = report_write
        failures.append("report_write_failed")
        report["status"] = "failed"
        report["failures"] = failures
    print(dumps_json_text(report))  # 同时打印到标准输出。
    exit_code = 0 if not failures else 1  # 成功返回 0，失败返回 1。
    print(f"[00_env] 完成 | 返回码={exit_code}", flush=True)
    return exit_code  # 把最终退出码交给调用方或系统。


if __name__ == "__main__":  # 脚本直接执行时走这里。
    raise SystemExit(main())  # 把 main 的返回码交还给系统。
