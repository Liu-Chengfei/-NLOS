"""场景构建入口聚合模块。

这个模块只负责把场景相关的常用函数统一导出，方便外部从 `liquidloc.scenarios` 直接导入。
它本身不实现具体场景逻辑，所有真正的规则都在子模块里。
这里导出的函数分别对应异步扰动、模态缺失、NLOS、视觉退化、几何布局和场景采样。
"""

from liquidloc.scenarios.async_levels import apply_async_level  # 导入异步扰动应用函数，用于对事件序列施加时间偏移、抖动和 blackout。
from liquidloc.scenarios.geometry_levels import build_anchor_layout  # 导入锚点布局构建函数，用于根据几何等级生成 anchor 坐标和报告。
from liquidloc.scenarios.missing_modalities import apply_modality_drop  # 导入模态缺失应用函数，用于按时间段屏蔽指定模态的事件。
from liquidloc.scenarios.nlos_levels import apply_nlos_level  # 导入 NLOS 等级应用函数，用于对 UWB 事件施加非视距偏差和质量退化。
from liquidloc.scenarios.scene_sampler import sample_scenes  # 导入场景采样函数，用于将实验配置展开为可执行的场景任务列表。
from liquidloc.scenarios.visual_levels import apply_visual_level  # 导入视觉退化应用函数，用于对 VIO 事件施加特征丢失、重投影误差和漂移。
from liquidloc.scenarios._event_utils import clone_event, get_event_value  # 导入事件克隆与取值工具，供场景子模块共享。

__all__ = (
    "apply_async_level",  # 导出异步扰动入口，供上层直接调用。
    "apply_modality_drop",  # 导出模态缺失入口，供 blackout 类实验使用。
    "apply_nlos_level",  # 导出 NLOS 入口，供 UWB 弱视距场景使用。
    "apply_visual_level",  # 导出视觉退化入口，供 VIO 退化实验使用。
    "build_anchor_layout",  # 导出几何布局入口，供 anchor 布局任务使用。
    "clone_event",  # 导出事件深拷贝工具。
    "get_event_value",  # 导出事件字段取值工具。
    "sample_scenes",  # 导出场景采样入口，供实验编排层使用。
)
