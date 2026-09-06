"""estimators 包的统一导出入口。

这个文件不承载具体算法，只负责把 EKF、FGO 以及状态定义相关的公共
符号集中导出，方便上层 pipeline、script 和测试直接从一个位置导入。
这里导出的内容必须和下游实际消费的一致，不能随意增删顺序或改名。
"""

from liquidloc.estimators.ekf_core import EKFCore  # 导出标准 EKF 核心实现，供上层直接使用。
from liquidloc.estimators.estimator_api import EstimatorAPI  # 导出估计器抽象基类，权威定义位于 estimators 层。
from liquidloc.estimators.fgo_core import FGOCore  # 导出滑窗 FGO 核心实现，供上层直接使用。
from liquidloc.estimators.robust_ekf_core import RobustEKFCore  # 导出带门控和鲁棒权重的 EKF 核心实现。
from liquidloc.estimators.shared import build_controlled_measurement_cov, control_to_dict  # 导出估计器共享工具函数。
from liquidloc.estimators.state_definition import (  # 导出状态定义相关的固定元数据，给下游统一取用。
    cov_dim,  # 协方差维度，通常和状态维度一致。
    get_state_index_map,  # 生成状态名到索引的映射。
    noise_dim,  # 噪声维度，供预测和更新步骤对齐。
    state_dim,  # 状态维度，供矩阵构造使用。
    state_index_map,  # 冻结后的状态索引映射，避免运行时被改写。
    state_items,  # 冻结的状态项顺序，是整个估计链的基础约定。
)

__all__ = (  # 只暴露这些符号，避免把内部工具函数误导成公开 API。
    "EKFCore",  # 标准 EKF 核心。
    "EstimatorAPI",  # 估计器抽象基类。
    "FGOCore",  # 滑窗 FGO 核心。
    "RobustEKFCore",  # 鲁棒 EKF 核心。
    "build_controlled_measurement_cov",  # 共享协方差控制函数。
    "control_to_dict",  # 共享控制对象转字典函数。
    "cov_dim",  # 协方差维度。
    "get_state_index_map",  # 状态索引映射生成函数。
    "noise_dim",  # 噪声维度。
    "state_dim",  # 状态维度。
    "state_index_map",  # 冻结状态映射。
    "state_items",  # 冻结状态顺序。
)
