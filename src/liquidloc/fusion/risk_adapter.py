"""风险适配器——兼容层包装，统一委托协议层做 risk 归一化。

本模块的职责：
1. 维持 fusion 侧旧接口 `(normalized_risk, risk_report)`
2. 对输入做最小兼容校验（拒绝布尔、拒绝非数值）
3. 把真正的风险裁剪逻辑委托给协议层唯一权威实现

核心概念：
- **风险值（risk）**：模型输出的标量信号，反映当前测量或场景的风险程度，
  取值范围由 `BRIDGE_THRESHOLDS` 中的 risk_min / risk_max 定义
- **裁剪（clipping）**：将超出安全范围的 risk 值强制限制到边界，
  防止极端 risk 值导致估计器行为异常

分层边界：
- 本模块属于 **fusion 桥接层**，只做"原始 risk → 安全 risk"的数值裁剪
- 不决定 risk 值本身（由模型推理产生），也不决定 risk 如何被使用
  （由桥接合约通过 noise_multiplier 消费）

与其他模块的关系：
- `protocol.liquid_bridge_contract.normalize_risk`：risk 裁剪的唯一权威实现
- `fusion_runner`：主实验链直接调用协议层；本模块仅保留兼容接口
"""

from __future__ import annotations  # 延迟解析注解，减少导入阶段的类型依赖。

from liquidloc.common.validation import is_bool_like  # 集中判断 bool / np.bool_。
from liquidloc.protocol.liquid_bridge_contract import normalize_risk as _normalize_risk_contract


def normalize_risk(raw_risk):
    """将原始风险值归一化到协议定义的合法范围。

    处理流程：
    1. 拒绝布尔输入（布尔值作为风险值毫无物理意义）
    2. 将输入转换为浮点数
    3. 把真正的归一化裁剪交给协议层权威实现
    4. 返回标准化结果和诊断报告

    Args:
        raw_risk: 模型输出的原始风险值，必须是可转换为有限浮点数的类型（bool 除外）。

    Returns:
        二元组 (normalized_risk, risk_report)：
        - normalized_risk: 裁剪后的风险值（float）
        - risk_report: 诊断报告字典，包含：
            - "raw_risk": 原始风险值
            - "clipped_risk": 裁剪后的风险值
            - "normalized_risk": 最终返回的标准化风险值（与 clipped_risk 相同）

    Raises:
        TypeError: 当输入是布尔值或无法转换为浮点数时。
        ValueError: 当输入是 NaN 或 inf 时。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"raw_risk": raw_risk, "input_type": type(raw_risk).__name__}, "normalize_risk")
    if is_bool_like(raw_risk):  # 布尔值不算合法风险数值（含 numpy 布尔）。
        raise TypeError("raw_risk must be a real number.")  # 直接拒绝布尔输入。
    try:  # 先尝试把输入转成 float。
        raw_risk = float(raw_risk)  # 只要能转成浮点数，就当作风险值处理。
    except (TypeError, ValueError) as exc:  # 转换失败说明不是数值。
        raise TypeError("raw_risk must be a real number.") from exc  # 统一抛类型错。
    normalized_risk = _normalize_risk_contract(raw_risk)  # 委托协议层做唯一的风险裁剪与有限性校验。
    risk_report = {  # 报告字典记录裁剪过程中的关键值。
        "raw_risk": raw_risk,  # 原始风险值。
        "clipped_risk": normalized_risk,  # 裁剪后的风险值。
        "normalized_risk": normalized_risk,  # 最终返回的标准化风险值。
    }  # 报告字典组装完毕。
    return normalized_risk, risk_report  # 同时返回标准化结果和报告。
