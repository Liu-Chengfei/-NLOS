"""输出契约检查模块。

文件职责：
  检查输出目录结构是否完整，即必需的子目录和文件是否都已生成。
  只做结构性检查，不检查文件内容，适合 smoke run、脚本验收和流水线结束后的快速验收。

本文件绝对不负责：
  不检查文件内容是否正确。
  不检查指标值是否合规。
  不修改任何文件或目录。

上游依赖：
  liquidloc.common.constants（DEFAULT_OUTPUT_DIRS 必需目录列表、
  DEFAULT_REQUIRED_OUTPUT_FILES 必需文件列表）

下游调用者：
  pipelines/contract_smoke_pipeline.py（流水线结束后的自检）、
  scripts/15_audit_outputs.py（输出验收脚本）

已知边界：
  本模块位于 protocol 层，但 check_output_contract 直接访问文件系统
  （Path.is_dir / Path.is_file / Path.resolve），属于协议验证的执行层
  而非协议定义层。契约清单（DEFAULT_OUTPUT_DIRS / DEFAULT_REQUIRED_OUTPUT_FILES）
  定义在 common/constants.py，本模块仅消费不修改。

输入对象定义：
  - output_root  输出根目录路径（str 或 Path）

输出对象定义：
  - check_output_contract  返回结构检查摘要字典

核心变量定义：
  无模块级变量，所有配置来自 constants。

关键设计决策：
  - 只检查目录和文件是否存在，不读内容，保证检查速度极快。
  - 返回值包含缺失项明细，便于下游精确定位问题。
"""

from __future__ import annotations  # 允许类型标注在后续扩展时保持灵活。

from pathlib import Path  # 用于处理输出根目录路径。
from typing import Any  # 允许类型注解里表示"任意类型"。

from liquidloc.common.constants import (  # 必需目录/文件清单（单源真相）。
    DEFAULT_OUTPUT_DIRS,
    DEFAULT_REQUIRED_OUTPUT_FILES,
    PUBLIC_BENCHMARK_OUTPUT_DIRS,
    PUBLIC_BENCHMARK_REQUIRED_OUTPUT_FILES,
)
from liquidloc.common.validation import is_string_like  # 字符串类型检查（含 numpy.str_）。
from liquidloc.protocol.version import OUTPUT_CONTRACT_VERSION  # 输出契约版本号。


# 公开合同布局名常量（单源真相，D15 漂移根因修复）。
LAYOUT_DEFAULT = "default"  # 默认 flat 布局（core pipeline / 训练 / 离线脚本）。
LAYOUT_PUBLIC_BENCHMARK = "public_benchmark"  # 公开 benchmark 嵌套布局（PublicBenchmarkPipeline）。


def check_output_contract(output_root: str | Path, *, layout: str = LAYOUT_DEFAULT) -> dict[str, Any]:
    """检查输出目录结构是否完整。

    扫描 output_root 下是否存在所有必需的子目录和文件，
    返回一个包含检查结果的摘要字典。

    参数：
        output_root：输出根目录路径，支持 str 或 Path 类型。
        layout：合同布局名；"default"（默认）使用 flat 布局（DEFAULT_OUTPUT_DIRS），
            "public_benchmark" 使用 PublicBenchmarkPipeline 的嵌套布局
            （PUBLIC_BENCHMARK_OUTPUT_DIRS / PUBLIC_BENCHMARK_REQUIRED_OUTPUT_FILES）。
            未知布局名抛 ValueError。

    返回：
        dict[str, Any]：结构检查摘要，包含以下键：
        - 'output_contract_version'：输出契约版本号。
        - 'output_root'：输出根目录的字符串表示。
        - 'layout'：实际使用的布局名。
        - 'required_dirs'：必需目录列表。
        - 'required_files'：必需文件列表。
        - 'missing_dirs'：缺失目录列表。
        - 'missing_files'：缺失文件列表。
        - 'is_complete'：布尔值，True 表示所有必需项都存在。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"output_root": str(output_root), "layout": layout}, "check_output_contract 入口参数")
    if not (is_string_like(output_root) or isinstance(output_root, Path)):  # output_root 必须是字符串类型或 Path（is_string_like 兼容 numpy.str_）。
        raise TypeError(f"output_root must be a string or Path, got {type(output_root).__name__}")
    if is_string_like(output_root) and not str(output_root).strip():  # 空白字符串不允许（is_string_like 兼容 numpy.str_）。
        raise ValueError("output_root must be a non-empty string or Path")
    if not is_string_like(layout):  # layout 必须是字符串。
        raise TypeError(f"layout must be a string, got {type(layout).__name__}")
    layout_name = str(layout).strip()  # 规整布局名。
    if layout_name == LAYOUT_DEFAULT:  # 默认 flat 布局。
        required_dirs = list(DEFAULT_OUTPUT_DIRS)
        required_files = list(DEFAULT_REQUIRED_OUTPUT_FILES)
    elif layout_name == LAYOUT_PUBLIC_BENCHMARK:  # 公开 benchmark 嵌套布局。
        required_dirs = list(PUBLIC_BENCHMARK_OUTPUT_DIRS)
        required_files = list(PUBLIC_BENCHMARK_REQUIRED_OUTPUT_FILES)
    else:  # 未知布局名直接拒绝。
        raise ValueError(
            f"unknown layout {layout_name!r}; expected one of {LAYOUT_DEFAULT!r} or {LAYOUT_PUBLIC_BENCHMARK!r}"
        )
    root = Path(output_root)  # 将输入统一转成 Path。
    # Path("") 等价于 Path(".")，会被解析为当前工作目录，属于非预期行为。
    if str(root).strip() in (".", ""):
        raise ValueError("output_root must not resolve to the current working directory")
    root = root.resolve()  # resolve() 规范化路径，消除符号链接和 .. 组件。
    # 注意：output_root 允许在项目根目录外（如临时目录、外部存储），
    # 不做 is_relative_to(project_root) 限制，与协议配置路径的严格限制不同。

    missing_dirs = [name for name in required_dirs if not (root / name).is_dir()]  # 找出缺失目录。
    missing_files = [name for name in required_files if not (root / name).is_file()]  # 找出缺失文件。

    return {  # 返回检查摘要。
        'output_contract_version': OUTPUT_CONTRACT_VERSION,  # 输出契约版本号，变更时表示结构不兼容。
        'output_root': str(root),  # 输出根目录。
        'layout': layout_name,  # 实际使用的布局名。
        'required_dirs': required_dirs,  # 必需目录列表。
        'required_files': required_files,  # 必需文件列表。
        'missing_dirs': missing_dirs,  # 缺失目录列表。
        'missing_files': missing_files,  # 缺失文件列表。
        'is_complete': not missing_dirs and not missing_files,  # 是否完整。
    }
