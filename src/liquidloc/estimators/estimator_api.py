"""估计器抽象接口合同。

职责：
    定义所有状态估计器（state estimator）必须遵守的抽象合同。
    估计器负责逐步消费事件并输出当前状态估计，是融合和流水线的
    核心计算单元。每个具体估计器都必须继承 EstimatorAPI
    并实现 reset、step 和 get_state 三个方法。

    本模块是 EstimatorAPI 的权威定义位置（原定义从 interfaces/estimator_api.py
    迁移至此，消除 estimators → interfaces 的逆向依赖）。

上游依赖：
    - Python 标准库 abc              — 提供抽象基类和抽象方法装饰器
    - Python 标准库 typing           — 提供 Any 类型，用于灵活标注输入事件类型
    - liquidloc.common.types         — 提供 StateEstimate 类型，作为估计方法的返回类型

下游调用者：
    - liquidloc.estimators.*         — 各具体估计器实现类继承 EstimatorAPI
    - liquidloc.interfaces.estimator_api — 对外统一导出入口（re-export）
    - liquidloc.pipelines.*          — 流水线层调用估计器的 step / get_state 驱动推理
    - liquidloc.fusion.*             — 融合层调用估计器获取状态估计

核心变量：
    - EstimatorAPI  — 估计器抽象基类，定义 reset / step / get_state 三个方法
"""

from __future__ import annotations  # 允许本文件内的类型注解使用前向引用，避免循环导入。

from abc import ABC, abstractmethod  # ABC 用于定义抽象基类，abstractmethod 用于标记必须由子类实现的方法。
from typing import Any  # Any 用于标注 event 参数，因为不同估计器可能接收不同结构的事件数据。

from liquidloc.common.types import StateEstimate  # StateEstimate 是状态估计摘要类型，包含 state、covariance_diag、timestamp。

__all__ = ("EstimatorAPI",)  # 明确本模块对外只导出 EstimatorAPI 一个名字。


class EstimatorAPI(ABC):  # 继承 ABC 使其成为抽象基类，不能直接实例化。
    """所有状态估计器的公共抽象合同。

    状态估计器负责逐步消费传感器事件（IMU/UWB/VIO 等），
    维护并更新内部状态估计（位置、速度、偏置等），
    并在需要时返回当前状态的快照。
    任何具体估计器（如扩展卡尔曼滤波器、无迹卡尔曼滤波器等）
    都必须继承此类并实现 reset、step 和 get_state 方法。

    上游：
        - 被具体估计器类继承（如 EKF、UKF 等）
    下游：
        - 融合器调用 step() 逐步推进状态估计
        - 流水线调用 get_state() 获取当前估计快照
        - 调度器调用 reset() 在新序列开始时重置估计器
    """

    @abstractmethod  # 标记为抽象方法，子类必须实现，否则子类也无法实例化。
    def reset(self, initial_state: dict | None = None) -> None:
        """重置估计器内部状态。

        在以下场景中必须调用此方法：
        - 切换到新序列时，需要将估计器恢复到初始状态
        - 重新开始估计时，需要清除上一轮的累积状态
        - 可选地传入初始状态字典，用于热启动估计器

        Args:
            initial_state (dict | None): 初始状态字典，键为状态字段名（如 "x"、"y"、"yaw"），
                值为对应的初始值。默认为 None，表示使用估计器内部的默认初始状态
                （通常为零向量）。

        Returns:
            None: 此方法无返回值，仅产生副作用（重置内部状态）。

        Raises:
            NotImplementedError: 如果子类未实现此方法（由 ABC 机制保证）
        """
        from liquidloc.common.tee_logger import print_dict  # 局部导入参数打印工具。
        print_dict({"initial_state": initial_state}, "reset 入参", prefix="[配置]")
        ...  # 省略号占位，表示此方法由子类实现，基类不提供默认行为。

    @abstractmethod  # 标记为抽象方法，子类必须实现，否则子类也无法实例化。
    def step(self, event: Any) -> StateEstimate:
        """消费一个事件并返回当前状态估计。

        这是估计器的核心推进方法。每调用一次，估计器内部会：
        1. 解析输入事件（如 IMU 测量、UWB 距离、VIO 位姿等）
        2. 执行预测步骤（根据运动模型预测下一时刻状态）
        3. 执行更新步骤（根据观测模型修正状态估计）
        4. 返回更新后的状态估计快照

        Args:
            event (Any): 输入事件数据。类型为 Any 是因为不同估计器
                可能接收不同结构的事件（如 IMU 数据包、UWB 测量、
                VIO 位姿等）。事件通常包含时间戳、模态标识和载荷数据。

        Returns:
            StateEstimate: 当前状态估计摘要，包含：
                - state (dict[str, float]): 状态向量，键为字段名（如 "x"、"y"、"yaw"）
                - covariance_diag (list[float]): 协方差矩阵对角线元素，反映各状态分量的不确定性
                - timestamp (float | None): 对应时间戳

        Raises:
            NotImplementedError: 如果子类未实现此方法（由 ABC 机制保证）

        §10.1(c) / §细节「在线约束」：step 仅消费 t 及以前到达的事件。
        禁止把整段已采集 bag 当未来已知再优化（禁止 post-hoc smoothing /
        full-batch trajectory 优化）。任何违反此约束的批量入口必须
        fail-loud 拒绝——不允许任何 estimator 路径绕过在线约束。
        """
        from liquidloc.common.tee_logger import print_dict  # 局部导入参数打印工具。
        print_dict({"event": event}, "step 入参", prefix="[配置]")
        ...  # 省略号占位，表示此方法由子类实现，基类不提供默认行为。

    @abstractmethod  # 标记为抽象方法，子类必须实现，否则子类也无法实例化。
    def get_state(self) -> StateEstimate:
        """返回当前状态估计的快照。

        此方法不推进估计器状态，仅读取当前内部状态的快照。
        适用于需要在不消费新事件的情况下查询当前估计的场景，
        例如在流水线结束时获取最终状态、或在调试时检查中间状态。

        Returns:
            StateEstimate: 当前状态估计摘要，包含：
                - state (dict[str, float]): 状态向量，键为字段名（如 "x"、"y"、"yaw"）
                - covariance_diag (list[float]): 协方差矩阵对角线元素，反映各状态分量的不确定性
                - timestamp (float | None): 最近一次更新对应的时间戳

        Raises:
            NotImplementedError: 如果子类未实现此方法（由 ABC 机制保证）
        """
        ...  # 省略号占位，表示此方法由子类实现，基类不提供默认行为。
