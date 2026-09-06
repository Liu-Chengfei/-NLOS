"""安全模式兼容壳——显式转发到协议层唯一安全模式实现。

本模块的职责：
1. 保留 fusion 侧历史入口 `apply_safe_mode`
2. 显式转发到协议层 `adjust_intermediate_for_safe_mode`
3. 不在桥接层复制第二份安全模式语义

核心概念：
- **安全模式（safe mode）**：当检测到异常场景（如传感器退化、环境剧变）时，
  将模型输出的激进控制量收缩到更保守的范围，防止估计器状态被异常控制量破坏
- **收缩策略**：由协议层 `liquid_bridge_contract.adjust_intermediate_for_safe_mode` 定义，
  本模块只负责转发调用

分层边界：
- 本模块属于 **fusion 桥接层**，只做"转发调用"，不实现安全逻辑
- 安全逻辑的权威实现位于 `protocol.liquid_bridge_contract`
- 本模块不得定义新的收缩规则或覆盖协议层的行为

与其他模块的关系：
- `protocol.liquid_bridge_contract.adjust_intermediate_for_safe_mode`：
  协议层的安全模式收缩函数，是本模块转发的目标
- `fusion_runner`：融合主循环，直接调用 `build_measurement_control`，
  后者内部直接调用协议层 `adjust_intermediate_for_safe_mode`，
  并不经过本兼容入口；本入口仅作为历史 API 保留，供外部直接调用
"""

from __future__ import annotations  # 延迟解析注解，减少导入阶段的类型依赖。

from typing import TYPE_CHECKING, Any

from liquidloc.protocol.liquid_bridge_contract import (
    adjust_intermediate_for_safe_mode as _adjust_intermediate_for_safe_mode,
)

if TYPE_CHECKING:  # 仅用于类型注解，不在运行时导入，避免无谓的依赖开销。
    from liquidloc.common.types import ModelIntermediate
    from liquidloc.protocol.scene_schema import SceneSpec


def apply_safe_mode(
    intermediate_outputs: "ModelIntermediate | dict[str, Any]",
    scene_context: "SceneSpec | str | dict[str, Any]",
    safe_mode_cfg: "dict[str, Any] | None" = None,
) -> "tuple[ModelIntermediate, bool]":  # 保留历史入口名，内部显式委托协议层。
    """将中间控制量收缩到更保守的安全范围。

    本函数是协议层安全模式逻辑的融合侧入口，直接转发给
    `adjust_intermediate_for_safe_mode`，不重新实现安全逻辑。
    这样做的好处是：
    - 安全模式的收缩规则只在协议层维护一份，避免多份实现产生不一致
    - 融合层只需关心"何时调用安全模式"，不需要关心"安全模式怎么做"

    Args:
        intermediate_outputs: 模型或融合链路给出的中间控制量，
            通常为 ModelIntermediate 实例或包含 bias/risk/uwb_scaling/vio_scaling 的字典。
        scene_context: 当前场景上下文，决定哪些控制项需要收紧；
            支持 SceneSpec、场景编码字符串或字典。
        safe_mode_cfg: 安全模式配置，决定是否启用收敛及风险阈值；
            读取 ``enabled``（bool）和 ``risk_threshold``（float，默认 0.5）两个字段，
            默认 ``None`` 等价于 ``{"enabled": False}``。

    Returns:
        二元组 ``(adjusted_outputs, converge_flag)``：
        - ``adjusted_outputs``：收缩后的 ``ModelIntermediate`` 实例（无论输入是对象还是字典）。
        - ``converge_flag``：是否触发了安全模式收敛（``bool``）。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "scene_context_type": type(scene_context).__name__,
        "safe_mode_cfg": safe_mode_cfg,
        "intermediate_type": type(intermediate_outputs).__name__,
    }, "apply_safe_mode")
    return _adjust_intermediate_for_safe_mode(  # 这里直接转发给协议层实现，不重新写安全逻辑。
        intermediate_outputs,  # 模型或融合链路给出的中间控制量。
        scene_context,  # 当前场景上下文，决定哪些控制项需要收紧。
        safe_mode_cfg,  # 安全模式配置，决定收缩策略和强度。
    )  # 转发完成后直接返回协议层的收缩结果。
