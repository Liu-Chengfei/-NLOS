"""模型后端抽象接口合同。

职责：
    定义所有模型后端（model backend）必须遵守的抽象合同。
    模型后端负责从特征窗口推断中间控制量（bias、risk、scaling），
    供下游估计器或融合器使用。每个具体模型都必须继承 ModelAPI
    并实现 reset 和 infer_intermediate 方法。

上游依赖：
    - Python 标准库 abc              — 提供抽象基类和抽象方法装饰器
    - Python 标准库 typing           — 提供 Any 类型，用于灵活标注输入张量类型
    - liquidloc.common.types         — 提供 ModelIntermediate 类型，作为推断方法的返回类型

下游调用者：
    - liquidloc.models.*             — 各具体模型实现类继承 ModelAPI
    - liquidloc.interfaces.__init__  — 统一导出 ModelAPI 给外部使用
    - liquidloc.estimators.*         — 估计器层调用模型的 infer_intermediate 获取中间控制量

核心变量：
    - ModelAPI  — 模型抽象基类，定义 reset / infer_intermediate 两个方法
"""

from __future__ import annotations  # 允许本文件内的类型注解使用前向引用，避免循环导入。

from abc import ABC, abstractmethod  # ABC 用于定义抽象基类，abstractmethod 用于标记必须由子类实现的方法。
from typing import Any  # Any 用于标注 window_tensor 参数；本接口为抽象合同，不耦合具体张量框架，具体输入类型由子类实现决定（与 infer_intermediate docstring 同口径）。

from liquidloc.common.types import ModelIntermediate  # ModelIntermediate 是模型中间态容器，包含 bias、risk、uwb_scaling、vio_scaling。

__all__ = ("ModelAPI",)  # 明确本模块对外只导出 ModelAPI 一个名字。


class ModelAPI(ABC):  # 继承 ABC 使其成为抽象基类，不能直接实例化。
    """所有模型后端的公共抽象合同。

    模型后端负责从输入特征窗口推断出中间控制量（偏置、风险、缩放系数），
    这些控制量会被下游的估计器或融合器用来调整状态估计过程。
    任何具体模型（如神经网络、规则模型等）都必须继承此类
    并实现 reset 和 infer_intermediate 方法。

    上游：
        - 被具体模型类继承
    下游：
        - 估计器 / 融合器调用 infer_intermediate() 获取中间控制量
        - 调度器调用 reset() 在新序列开始时重置模型状态
    """

    @abstractmethod  # 标记为抽象方法，子类必须实现，否则子类也无法实例化。
    def reset(self) -> None:
        """重置模型内部状态，确保下次推理起始于干净状态。

        在以下场景中必须调用此方法：
        - 切换到新序列时，按需清除上一序列遗留的内部状态
        - 重新开始推理时，需要将模型恢复到初始状态

        Returns:
            None: 此方法无返回值，仅产生副作用（重置内部状态）。

        Raises:
            TypeError: 子类未实现 reset 时，由 ABC 机制在实例化该子类时抛出
                （本抽象方法体为占位，调用本身不抛异常）。
        """
        ...  # 省略号占位，表示此方法由子类实现，基类不提供默认行为。

    @abstractmethod  # 标记为抽象方法，子类必须实现，否则子类也无法实例化。
    def infer_intermediate(self, window_tensor: Any) -> ModelIntermediate:
        """从一个特征窗口推断一组中间控制量。

        这是模型后端的核心推理方法。接收一个特征窗口，
        输出包含偏置、风险和模态缩放系数的中间态容器。四头输出合同
        （bias/risk/uwb_scaling/vio_scaling）的数量、字段名和字段含义
        保持冻结，对齐 docs/liquid_architecture_current.md §2.2。

        Args:
            window_tensor (Any): 输入特征窗口。类型标注为 Any 是因为本接口
                为抽象合同，不耦合具体张量框架；具体输入类型由子类实现决定。
                窗口通常包含一段时间步的 IMU/UWB/VIO 多模态特征。

        Returns:
            ModelIntermediate: 模型中间态容器。四头字段语义对齐
                docs/liquid_architecture_current.md §2.3，值域约束由
                ModelIntermediate.__post_init__ 强制（单源真相常量定义于
                liquidloc.common.constants）：
                - bias (float): UWB 测量域偏置修正候选量，中性值 0.0，
                  非负，范围 [0.0, BRIDGE_BIAS_MAX]
                - risk (float): pre-bridge 基础风险（对齐风险），中性值 0.0，
                  范围 [BRIDGE_RISK_MIN, BRIDGE_RISK_MAX]
                - uwb_scaling (float): UWB 模态额外观测噪声膨胀系数，
                  中性值 1.0（不膨胀），范围 [1.0, BRIDGE_SCALING_MAX]
                - vio_scaling (float): VIO 模态额外观测噪声膨胀系数，
                  中性值 1.0（不膨胀），范围 [1.0, BRIDGE_SCALING_MAX]

        Raises:
            ValueError: 当模型输出不满足四头合同（输出头数量、有限性、
                值域约束）时由具体实现抛出。
            RuntimeError: 当风险校准或张量设备迁移失败时由具体实现抛出。
        """
        from liquidloc.common.tee_logger import print_dict
        print_dict({"window_tensor_type": type(window_tensor).__name__}, "ModelAPI.infer_intermediate 入口参数")
        ...  # 省略号占位，表示此方法由子类实现，基类不提供默认行为。
