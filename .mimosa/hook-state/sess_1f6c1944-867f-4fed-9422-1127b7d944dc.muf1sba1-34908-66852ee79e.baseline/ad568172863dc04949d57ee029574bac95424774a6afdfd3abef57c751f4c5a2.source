"""风险映射协议层权威定义。

【文件职责】
登记风险值线性映射公式 ``risk = risk * (risk_max - risk_min) + risk_min``，
把已归一化到 [0, 1] 的风险投影到协议区间 [risk_min, risk_max]，并在末尾做
防御性 clamp。本文件是该公式的唯一权威实现，其他层（inference / trainer /
factory）禁止重实现该公式，应统一引用本函数。

【科学真相】
当 ``risk_min = 0.0`` 且 ``risk_max = 1.0``（默认协议区间）时：
  - ``risk_span = 1.0``，线性映射分支被 ``if risk_span != 1.0 or risk_min != 0.0``
    条件跳过，公式退化为恒等映射，数值不变；
  - clamp ``min(1.0, max(0.0, risk))`` 对已落在 [0, 1] 的输入同样不改变数值。
因此当前默认配置下整体为恒等映射，仅协议可追溯性提升。

当协议区间非默认（例如 ``risk_min=0.2, risk_max=0.8``）时：
  - 先做线性映射 ``risk = risk * 0.6 + 0.2``；
  - 再 clamp 到 [0.2, 0.8] 防止浮点漂移越界。

【分层边界】
本函数只负责标量 float 的公式实现；训练侧张量版本（``_project_train_risk``）
因携带 sigmoid 预归一化与 ``torch.clamp`` 张量语义，口径与本函数不一致，不在
本文件统一，仅以本函数为公式锚点保持可追溯。

【上游依赖】
无（仅依赖 Python 标准库 math）。

【下游调用者】
models/liquid/inference.py（推理侧风险投影）、protocol/__init__.py（导出）。
"""

import math  # 用于有限性检查，防止 NaN/Inf 经 min/max 静默穿透。


def project_risk_to_protocol_range(risk, risk_min, risk_max):
    """把已归一化的风险值线性映射到协议区间 [risk_min, risk_max] 并 clamp。

    权威公式（协议层唯一实现，禁止其他层重实现）::

        risk_span = risk_max - risk_min
        if risk_span != 1.0 or risk_min != 0.0:
            risk = risk * risk_span + risk_min
        risk = min(risk_max, max(risk_min, risk))

    参数:
        risk: 已归一化到 [0, 1] 的风险值（任意可转 float 类型）。
        risk_min: 协议区间下界（取自 BRIDGE_THRESHOLDS["risk_min"]）。
        risk_max: 协议区间上界（取自 BRIDGE_THRESHOLDS["risk_max"]）。

    返回值:
        映射到 [risk_min, risk_max] 区间的风险值（float）。

    数值不变性:
        当 ``risk_min=0.0`` 且 ``risk_max=1.0`` 时为恒等映射，数值不变；
        当非默认区间时做线性映射后 clamp。

    D5 数值安全:
        入口做 float() 强制转换与 math.isfinite 检查，防止 NaN/Inf 经 min/max 会静默穿透（Python 中 max(0.0, nan) 返回 nan）。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"risk": risk, "risk_min": risk_min, "risk_max": risk_max}, "project_risk_to_protocol_range 入参", prefix="[配置]")
    risk = float(risk)  # D5：入口强制 float 转换，统一数值类型。
    if not math.isfinite(risk):  # D5：NaN/Inf 经 min/max 会静默穿透，必须显式拒绝。
        raise ValueError(f"risk must be finite, got {risk}")
    risk_span = risk_max - risk_min
    if risk_span != 1.0 or risk_min != 0.0:
        risk = risk * risk_span + risk_min
    return min(risk_max, max(risk_min, risk))
