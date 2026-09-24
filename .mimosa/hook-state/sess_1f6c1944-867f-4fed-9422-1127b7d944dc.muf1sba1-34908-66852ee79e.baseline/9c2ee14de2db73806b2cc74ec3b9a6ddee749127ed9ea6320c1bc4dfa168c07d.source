"""
文件：`src/liquidloc/plotting/__init__.py`

这个模块是 plotting 包的公共导出入口，负责把子模块里的构建函数和渲染函数
统一暴露给外部调用者。它本身不包含任何绘图逻辑，只做"重新导出"这一件事。

上游依赖：
- 各子模块（plot_calibration、plot_cases、plot_main_table、plot_runtime、
  plot_sweeps、plot_training_trends、plot_trajectories）定义了具体的
  build_*_spec 和 render_* 函数。

下游调用者：
- scripts/13_generate_figures.py 等脚本通过 `from liquidloc.plotting import ...`
  获取绘图入口。
- 测试文件通过 `from liquidloc.plotting import ...` 导入被测函数。

核心变量：
- __all__：显式声明本包对外暴露的公开名称列表，防止 `from liquidloc.plotting import *`
  时意外导入内部符号。
"""

# 从校准图子模块导入构建函数和渲染函数，用于渲染 calibration 相关的 SVG 图。
from liquidloc.plotting.plot_calibration import build_calibration_figure_spec, render_calibration_figure
# 从案例图子模块导入构建函数和渲染函数，用于把 main/failure/boundary 三组案例渲染成文本优先图片。
from liquidloc.plotting.plot_cases import build_case_figure_manifest, render_case_figures
# 从主表图子模块导入构建函数和渲染函数，用于把主表数据渲染成可视化表格图片。
from liquidloc.plotting.plot_main_table import build_main_table_figure_spec, render_main_table_figure
# 从运行时图子模块导入构建函数和渲染函数，用于把 runtime 指标渲染成 SVG 面板图。
from liquidloc.plotting.plot_runtime import build_runtime_figure_spec, render_runtime_figure
# 从 sweep 图子模块导入构建函数和渲染函数，用于把 sweep 表渲染成折线或柱状 SVG 图。
from liquidloc.plotting.plot_sweeps import build_sweep_figure_spec, render_sweep_figure
# 从训练趋势图子模块导入构建函数和渲染函数，用于把训练趋势报告渲染成 SVG 图。
from liquidloc.plotting.plot_training_trends import (
    build_training_trend_figure_spec,  # 构建训练趋势图的规范对象。
    render_training_trend_figure,  # 渲染训练趋势图并写入磁盘。
)
# 从轨迹图子模块导入构建函数和渲染函数，用于把预测/真实轨迹渲染成 2D/3D 对比图。
from liquidloc.plotting.plot_trajectories import build_trajectory_figure_spec, render_trajectory_figure

# __all__ 显式列出本包的公开 API，保证 from liquidloc.plotting import * 只导出这些名称。
__all__ = (
    "build_calibration_figure_spec",  # 校准图规范构建函数。
    "build_case_figure_manifest",  # 案例图清单登记函数。
    "build_main_table_figure_spec",  # 主表图规范构建函数。
    "build_runtime_figure_spec",  # 运行时图规范构建函数。
    "build_sweep_figure_spec",  # sweep 图规范构建函数。
    "build_training_trend_figure_spec",  # 训练趋势图规范构建函数。
    "build_trajectory_figure_spec",  # 轨迹图规范构建函数。
    "render_calibration_figure",  # 校准图渲染函数。
    "render_case_figures",  # 案例图渲染函数。
    "render_main_table_figure",  # 主表图渲染函数。
    "render_runtime_figure",  # 运行时图渲染函数。
    "render_sweep_figure",  # sweep 图渲染函数。
    "render_training_trend_figure",  # 训练趋势图渲染函数。
    "render_trajectory_figure",  # 轨迹图渲染函数。
)
