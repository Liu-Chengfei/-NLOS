"""桥接层/融合层业务阈值定义。

职责：
    集中放置桥接层（bridge）和融合层（fusion）使用的业务阈值，
    包括风险归一化范围、缩放约束、硬跳过门槛、偏置修正上限、噪声倍数封顶等。
    这些阈值属于桥接/融合层的业务规则，不属于 common 层纯算法/协议级常量。

上游依赖：
    - liquidloc.common.constants（BRIDGE_RISK_MIN / BRIDGE_RISK_MAX / BRIDGE_SCALING_MAX 单源真相）

下游调用者：
    - liquidloc.protocol.liquid_bridge_contract  — 桥接契约校验与噪声倍数计算
    - liquidloc.fusion.fusion_runner             — 融合层风险/缩放裁剪
    - liquidloc.factories.model_factory          — 模型输出风险映射
    - liquidloc.models.liquid.trainer            — Liquid 训练器风险裁剪
    - liquidloc.models.lstm.trainer              — LSTM 训练器风险裁剪
    - liquidloc.models.liquid.inference          — Liquid 推理风险映射
    - liquidloc.models.lstm.inference            — LSTM 推理风险映射
    - liquidloc.models.features.normalization    — 缩放归一化
    - liquidloc.estimators.fgo_core              — FGO 质量门槛
    - liquidloc.sensors.vision_model             — VIO 航向范围校验

核心变量：
    - BRIDGE_THRESHOLDS — 桥接层业务阈值字典
"""

from __future__ import annotations  # 保持类型注解写法一致。

from liquidloc.common.constants import (  # 桥接层阈值单源真相，protocol 层从此处消费。
    BRIDGE_RISK_MIN,
    BRIDGE_RISK_MAX,
    BRIDGE_SCALING_MAX,
    BRIDGE_BIAS_ABSOLUTE_MAX,
    BRIDGE_NOISE_MULTIPLIER_CEILING,
    BRIDGE_NON_CURRENT_SCALING_CEILING,
)


class _FrozenDict(dict):  # type: ignore[misc]
    """只读字典包装，禁止增删改操作。

    参考 event_schema.py 的同名实现，包装 BRIDGE_THRESHOLDS 防止运行时篡改阈值。
    所有现有读取代码（``BRIDGE_THRESHOLDS["key"]``、``BRIDGE_THRESHOLDS.get("key")``）
    均可通过 dict 接口正常访问，仅禁止写入/删除操作。
    """

    __slots__ = ()

    def __setitem__(self, key, value):  # type: ignore[override]
        raise TypeError("BRIDGE_THRESHOLDS is read-only")

    def __delitem__(self, key):  # type: ignore[override]
        raise TypeError("BRIDGE_THRESHOLDS is read-only")

    def pop(self, *args):  # type: ignore[override]
        raise TypeError("BRIDGE_THRESHOLDS is read-only")

    def popitem(self):  # type: ignore[override]
        raise TypeError("BRIDGE_THRESHOLDS is read-only")

    def clear(self):  # type: ignore[override]
        raise TypeError("BRIDGE_THRESHOLDS is read-only")

    def update(self, *args, **kwargs):  # type: ignore[override]
        raise TypeError("BRIDGE_THRESHOLDS is read-only")

    def setdefault(self, *args):  # type: ignore[override]
        raise TypeError("BRIDGE_THRESHOLDS is read-only")

    def __ior__(self, other):  # type: ignore[override]
        raise TypeError("BRIDGE_THRESHOLDS is read-only")

    def __hash__(self):  # type: ignore[override]
        return hash(frozenset(self.items()))

    def __eq__(self, other):  # type: ignore[override]
        if isinstance(other, _FrozenDict):
            return dict(self) == dict(other)
        return dict(self) == other

    def __contains__(self, key):  # type: ignore[override]
        return dict.__contains__(self, key)


_BRIDGE_THRESHOLDS_DATA = {  # 桥接层/融合层业务阈值表（原始可变 dict，由 _FrozenDict 包装为只读）。
    "scaling_min": 1.0,  # 缩放最小值（v3：回退到 1.0。v2 放宽到 0.5 的 soft-mask 在 e9 场景下让 Liquid 不当降权好测量，不利于高 NLOS + 异步场景。）
    "scaling_max": BRIDGE_SCALING_MAX,  # 缩放最大值（方差倍数，防止极端值导致数值溢出）。
    "non_current_scaling_ceiling": BRIDGE_NON_CURRENT_SCALING_CEILING,  # 非当前模态 scaling 上界（§12.3-C2a 三网同一写入口：第十八轮穷举自审修复，从 model_factory.py 硬编码 2.5 迁移至此单源）。
    "risk_min": BRIDGE_RISK_MIN,  # 风险最小值。
    "risk_max": BRIDGE_RISK_MAX,  # 风险最大值。
    "uwb_hard_skip_quality_floor": 0.10,  # UWB 质量硬跳过门槛，v2 放宽至 0.10 释放更多 UWB 更新。
    "vio_hard_skip_quality_floor": 0.12,  # VIO 质量硬跳过门槛，v2 放宽至 0.12 释放更多 VIO 更新。
    "risk_hard_skip_threshold": 1.05,  # 风险硬跳过阈值, v3 提高至 1.05 以匹配 risk_max=1.0, 完全解除硬跳过. 修复训练-推理风险饱和导致的 100% skip 问题 (诊断 2026-08-02).
    "uwb_bias_max_ratio": 0.5,  # UWB 偏置修正量占原始测距的最大比例。
    "uwb_bias_absolute_max": BRIDGE_BIAS_ABSOLUTE_MAX,  # UWB 偏置绝对值上限（米），v2 放宽至 5.0m。
    # ceiling 与 scaling_max 关系：ceiling = scaling_max² × (1+risk_max) = 50² × 2 = 5000。
    # D11-R2 修复后 ceiling=5000.0，scaling_max=50 在 risk>0 时不再被截断为死代码。
    # 与 liquid_bridge_contract._compose_noise_multiplier（noise_multiplier = scaling² × (1+risk)）一致。
    "uwb_noise_multiplier_ceiling": BRIDGE_NOISE_MULTIPLIER_CEILING,  # UWB 噪声倍数硬上限。
    "vio_noise_multiplier_ceiling": BRIDGE_NOISE_MULTIPLIER_CEILING,  # VIO 噪声倍数硬上限。
    "vio_dyaw_min": -6.283185307179586,  # VIO 航向变化下界（rad），约 -2π。
    "vio_dyaw_max": 6.283185307179586,  # VIO 航向变化上界（rad），约 +2π。
    "flow_missing_dyaw_quality_penalty": 0.5,  # flow 缺少 dyaw 时的质量惩罚值，将 quality 下调至此值以标记旋转信息不可靠。
    "imu_missing_inflation": 10.0,  # IMU 字段缺失时 process_noise 膨胀系数（§11.2 Q 固定：协议写死，非 estimator 私调）。
    "cov_jitter_eps": 1e-9,  # §11.5 SPD 抖动注入：当 Cholesky 失败时先 S += cov_jitter_eps * I 再重试；二次仍失败才 raise。
    "max_consecutive_skip_count": 1000.0,  # §19.1 永久拒识运行时拦截门限：连续跳过帧数超过此值即视为假第一。v4 放宽至 1000 以容纳 sim_e9 中长走廊序列 (避免误杀 valid long no-update runs)。
}  # 阈值表结束。

BRIDGE_THRESHOLDS = _FrozenDict(_BRIDGE_THRESHOLDS_DATA)  # D11-M2：包装为只读 _FrozenDict，禁止运行时篡改阈值。

# Chi-square 0.95 分位数冻结真相（IEEE 802.15.4a / MATLAB trackingEKF 对齐）。
# 协议层兜底真相：estimator YAML 中 gate.mahalanobis_sq 阈值必须与对应自由度的 chi2(0.95) 匹配，
# 此处提供权威数值供 estimator 配置校验和 NIS 门控兜底使用。
# 关键自由度的 0.95 分位数（来源：scipy.stats.chi2.ppf(0.95, df) / IEEE 802.15.4a 信道模型）：
#   - 1-DoF: 3.841459  (UWB 单锚测距残差)
#   - 3-DoF: 7.814725  (VIO 3D 位姿残差)
#   - 4-DoF: 9.487729  (UWB+VIO 联合 4 维残差)
#   - 6-DoF: 12.591587 (VIO 6D 位姿残差)
# 修改此字典必须同步检查 estimator YAML 中 gate.mahalanobis_sq 配置的一致性。
CHI2_95_PERCENTILES = _FrozenDict({
    1: 3.841459,   # UWB 1-DoF
    3: 7.814725,   # VIO 3-DoF
    4: 9.487729,
    6: 12.591587,
})

# 测量自由度默认映射（协议层兜底真相）。
# estimator YAML 中 gate.mahalanobis_sq 阈值应与此处对应自由度的 chi2(0.95) 匹配：
#   - uwb: 1-DoF → chi2(0.95,1)=3.841459
#   - vio: 3-DoF → chi2(0.95,3)=7.814725
# 此映射防止 estimator 配置中自由度与模态错配（如 UWB 误用 3-DoF 阈值）。
DEFAULT_GATING_DOF = _FrozenDict({"uwb": 1, "vio": 3})  # 测量自由度


def validate_bridge_thresholds() -> dict:  # 检查桥接阈值注册表内部一致性。
    """生成桥接阈值注册表检查报告。

    检查内容包括：
    1. BRIDGE_THRESHOLDS 中风险/缩放区间的最小值是否不超过最大值。
    2. 质量硬跳过门槛是否落在 [0, 1] 合理区间内。
    3. 风险硬跳过阈值是否落在 [risk_min, risk_max] 区间内。
    4. 正值约束：uwb_bias_absolute_max、噪声倍数上限必须为正。
    5. UWB 偏置比例必须在 (0, 1] 区间内。
    6. 缩放下限 scaling_min 必须 > 0 且 <= scaling_max（v2：允许 < 1.0 配合 soft-mask 释放调节空间）。

    Returns:
        dict: 包含以下键的检查报告字典：
            - required_names (list[str]): 应该存在的常量名列表。
            - missing_names (list[str]): 实际缺失的常量名列表。
            - contract_errors (list[str]): 契约错误描述列表。
            - is_complete (bool): 所有检查是否全部通过。
    """
    required_names = [  # 先列出这个模块应该存在的关键常量名。
        "BRIDGE_THRESHOLDS",  # 桥接阈值表。
    ]  # required_names 结束。
    missing_names = [name for name in required_names if name not in globals()]  # 找出当前模块里缺失的常量名。
    contract_errors = []  # 用这个列表收集契约错误。
    if not missing_names:  # 只有基础常量齐全时才继续做内部一致性检查。
        # 风险区间要保持有序。
        if BRIDGE_THRESHOLDS["risk_min"] > BRIDGE_THRESHOLDS["risk_max"]:  # 风险区间不能反着写。
            contract_errors.append(
                f"risk_thresholds_out_of_order: risk_min={BRIDGE_THRESHOLDS['risk_min']} > risk_max={BRIDGE_THRESHOLDS['risk_max']}"
            )
        # 缩放区间要保持有序。
        if BRIDGE_THRESHOLDS["scaling_min"] > BRIDGE_THRESHOLDS["scaling_max"]:  # 缩放区间不能反着写。
            contract_errors.append(
                f"scaling_thresholds_out_of_order: scaling_min={BRIDGE_THRESHOLDS['scaling_min']} > scaling_max={BRIDGE_THRESHOLDS['scaling_max']}"
            )
        # VIO 航向变化区间要保持有序。
        if BRIDGE_THRESHOLDS["vio_dyaw_min"] > BRIDGE_THRESHOLDS["vio_dyaw_max"]:  # VIO 航向变化区间不能反着写。
            contract_errors.append(
                f"vio_dyaw_thresholds_out_of_order: vio_dyaw_min={BRIDGE_THRESHOLDS['vio_dyaw_min']} > vio_dyaw_max={BRIDGE_THRESHOLDS['vio_dyaw_max']}"
            )
        # 质量硬跳过门槛必须落在 [0, 1] 合理区间内。
        if not (0.0 <= BRIDGE_THRESHOLDS["uwb_hard_skip_quality_floor"] <= 1.0):
            contract_errors.append(
                f"uwb_hard_skip_quality_floor_out_of_range: value={BRIDGE_THRESHOLDS['uwb_hard_skip_quality_floor']}, expected_range=[0.0, 1.0]"
            )
        if not (0.0 <= BRIDGE_THRESHOLDS["vio_hard_skip_quality_floor"] <= 1.0):
            contract_errors.append(
                f"vio_hard_skip_quality_floor_out_of_range: value={BRIDGE_THRESHOLDS['vio_hard_skip_quality_floor']}, expected_range=[0.0, 1.0]"
            )
        # 风险硬跳过阈值必须落在 [risk_min, risk_max] 区间内。
        _r_min = BRIDGE_THRESHOLDS["risk_min"]
        _r_max = BRIDGE_THRESHOLDS["risk_max"]
        if not (_r_min <= BRIDGE_THRESHOLDS["risk_hard_skip_threshold"] <= _r_max):
            contract_errors.append(
                f"risk_hard_skip_threshold_out_of_risk_range: value={BRIDGE_THRESHOLDS['risk_hard_skip_threshold']}, expected_range=[{_r_min}, {_r_max}]"
            )
        # 正值约束：这些阈值必须严格为正。
        if BRIDGE_THRESHOLDS["uwb_bias_absolute_max"] <= 0:
            contract_errors.append(f"uwb_bias_absolute_max_must_be_positive: value={BRIDGE_THRESHOLDS['uwb_bias_absolute_max']}")
        if BRIDGE_THRESHOLDS["uwb_noise_multiplier_ceiling"] <= 0:
            contract_errors.append(f"uwb_noise_multiplier_ceiling_must_be_positive: value={BRIDGE_THRESHOLDS['uwb_noise_multiplier_ceiling']}")
        if BRIDGE_THRESHOLDS["vio_noise_multiplier_ceiling"] <= 0:
            contract_errors.append(f"vio_noise_multiplier_ceiling_must_be_positive: value={BRIDGE_THRESHOLDS['vio_noise_multiplier_ceiling']}")
        # UWB 偏置比例必须在 (0, 1] 区间内。
        if not (0 < BRIDGE_THRESHOLDS["uwb_bias_max_ratio"] <= 1.0):
            contract_errors.append(f"uwb_bias_max_ratio_must_be_in_0_to_1: value={BRIDGE_THRESHOLDS['uwb_bias_max_ratio']}")
        # 缩放下限必须 > 0，v2 放宽至允许 <1.0 以配合 soft-mask。
        if BRIDGE_THRESHOLDS["scaling_min"] <= 0.0:
            contract_errors.append(f"scaling_min_must_be_positive: value={BRIDGE_THRESHOLDS['scaling_min']}")
        # flow 缺少 dyaw 时的质量惩罚值必须落在 [0, 1] 合理区间内。
        if not (0.0 <= BRIDGE_THRESHOLDS["flow_missing_dyaw_quality_penalty"] <= 1.0):
            contract_errors.append(
                f"flow_missing_dyaw_quality_penalty_out_of_range: value={BRIDGE_THRESHOLDS['flow_missing_dyaw_quality_penalty']}, expected_range=[0.0, 1.0]"
            )
    return {  # 返回一个统一的检查报告。
        "required_names": required_names,  # 需要存在的常量名。
        "missing_names": missing_names,  # 缺失的常量名。
        "contract_errors": contract_errors,  # 契约错误列表。
        "is_complete": not missing_names and not contract_errors,  # 是否完全通过检查。
    }  # 报告字典结束。


from liquidloc.common.tee_logger import print_dict  # 局部导入参数打印工具。

print_dict(  # 打印桥接层业务阈值常量，供审计核对。
    BRIDGE_THRESHOLDS,
    "bridge_thresholds.py 常量",
    prefix="[配置]",
)
