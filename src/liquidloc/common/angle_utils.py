"""二维平面角度工具。

职责：
    专门处理 yaw 和 dyaw 的归一化问题，保证角度比较不会因为跨越
    `-pi` / `pi` 边界而出现错误差值。

上游依赖：
    - Python 标准库 math             — 基础数学函数（fmod、pi）

下游调用者：
    - liquidloc.common.__init__      — 统一导出角度工具给上层
    - liquidloc.estimators.*         — 估计器层用角度工具做 yaw 归一化与角差计算
    - liquidloc.analysis.*           — 分析层用角度工具做指标中的角度比较

核心变量：
    - 无模块级变量，全部通过函数实现

MATLAB 对照：
    - wrap_angle_rad 区间为 [-pi, pi) 半开区间（pi 映射到 -pi）；
      MATLAB wrapToPi 区间为 (-pi, pi] 半开区间（pi 映射到自身）。
      两者仅在输入恰好为 pi 的奇数倍时映射方向不同，工程场景中几乎不触发。
    - angle_delta_rad 语义近似于 MATLAB wrapToPi(a - b)，区间约定同上。
"""

from __future__ import annotations  # 允许函数签名里使用前向引用类型。

import math  # 只需要基础数学函数。

from liquidloc.common.validation import coerce_finite_scalar


def wrap_angle_rad(value: float) -> float:  # 把任意弧度角包裹进标准区间。
    """把角度包裹到 ``[-pi, pi)`` 半开区间。

    近似于 MATLAB ``wrapToPi``，但区间约定不同：本函数为 ``[-pi, pi)``
    （pi 映射到 -pi），MATLAB ``wrapToPi`` 为 ``(-pi, pi]``（pi 映射到自身）。
    两者仅在输入恰好为 pi 的奇数倍时映射方向不同。

    利用 fmod 做余数运算，再通过偏移把结果归一化到标准区间。

    注意：本函数在非边界处导数恒为 1，EKF/FGO 雅可比矩阵中航向行
    取 +/-1 是正确的；但在航向差恰好跨越 +/-pi 边界时导数不连续，
    雅可比不再精确，这是所有基于线性化的角度处理方法的固有限制。

    Args:
        value (float): 任意弧度值，可以是任意实数。NaN 和 Inf 会被拒绝。

    Returns:
        float: 归一化后的弧度值，保证落在 [-pi, pi) 区间内。

    Raises:
        ValueError: 当 value 为 NaN 或 Inf 时抛出。
    """
    fv = coerce_finite_scalar(value, name="wrap_angle_rad value")  # NaN 和 Inf 无法归一化，必须尽早拒绝。
    # 先把值整体平移到以 0 为中心的正区间，便于做余数计算。
    wrapped = math.fmod(fv + math.pi, 2.0 * math.pi)
    # fmod 对负数会保留负号，所以这里要额外补回正区间。
    if wrapped < 0.0:
        # 再平移一个完整周期，把结果拉回目标区间。
        wrapped += 2.0 * math.pi
    # 最后再减回 pi，把区间落回 [-pi, pi)。
    return wrapped - math.pi


def angle_delta_rad(lhs: float, rhs: float) -> float:  # 计算两个角度之间的最短环形差。
    """计算两角之差并自动包裹到标准区间。

    近似于 MATLAB ``wrapToPi(lhs - rhs)``，区间约定同 wrap_angle_rad
    （``[-pi, pi)`` 半开区间）。先做原始差值（lhs - rhs），再归一化，
    保证返回值是最短环形差。

    Args:
        lhs (float): 左侧角度（弧度），通常为当前角度。
        rhs (float): 右侧角度（弧度），通常为参考角度。

    Returns:
        float: 两角之差的归一化结果，落在 [-pi, pi) 区间内。
    """
    # 先做原始差值，再交给 wrap 函数处理边界跳变。
    return wrap_angle_rad(float(lhs) - float(rhs))


__all__ = ("wrap_angle_rad", "angle_delta_rad")
