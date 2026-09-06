"""协议版本注册模块。

文件职责：
  集中保存几类冻结版本号，供校验、审计和 smoke check 统一引用。
  每个版本号对应一个协议层面的"不可变快照"，当协议发生不兼容变更时，
  必须在此处升级版本号，以便下游检测到版本漂移并中止。

本文件绝对不负责：
  不定义协议内容本身，只记录版本标识。

上游依赖：
  无外部依赖，本模块是协议层最底层的纯常量模块。

下游调用者：
  liquidloc.protocol.__init__（统一导出）、
  各协议子模块（版本号校验）、
  smoke check 脚本和审计脚本。

核心变量：
  PROTOCOL_VERSION：整体协议版本号，变更时表示协议格式不兼容。
  CONFIG_VERSION：配置文件格式版本号（预留），变更时表示配置结构不兼容。当前无 YAML 字段对应，待配置校验链路完善后启用。
  OUTPUT_CONTRACT_VERSION：输出契约版本号，变更时表示输出目录/文件结构不兼容。
  SNAPSHOT_VERSION：快照版本号，变更时表示实验快照格式不兼容。被 result_schema.ExperimentResult.to_dict() 写入实验结果。
"""

from __future__ import annotations  # 保持类型标注的扩展空间。

PROTOCOL_VERSION = 27  # v27 协议版本 (2026-09-02): 与《五轴档位协议定义.md》(H27 协议 v27) 对齐。
# v2 → v27 不兼容变更:
#   (1) K 轴：删除 K4（合并不再使用）；保留 K0/K1/K3（handbook S2）
#   (2) G 轴：4 档全部并入 K 轴（K 轴 = 原 G 轴 4 档 × 锚点数量组合）
#   (3) A 轴：A2/A3 为主档；A0/A1 仅用于 baseline 实验
#   (4) N 轴：N2/N3 为主档；N0/N1 仅用于 baseline
#   (5) V 轴：V0 为主档；V1/V2/V3 仅用于敏感性
#   (6) M 轴：M1 为主档（UWB 5% 成簇丢包，间隙 0.3–2s）
# v1 → v2 历史变更:
#   (1) perturbation_injection_point: sim_materializer → dual_track (D-1)
#   (2) 加 imu_jitter_exemption + step_dt_threshold (D-8)
#   (3) 加 scene_axis_diversity / geometry_axis_expansion / anchor_count_axis_expansion /
#       sensor_frequency_diversity / splits_ood_balance 五个扩展维度 (D-9 ~ D-13)
#   (4) SimSequenceSpec 加 axes_override + dt_imu_override / dt_uwb_override / dt_vio_override
#   (5) stable_window_start 加 selection_mode="poisson" 模式 (D-2)
#   (6) SimNoiseSpec 加 warn_missing_fixture_fields 审计 (D-7)
#   (7) visual_levels.py drift_bias_m ↔ drift_bias_sigma_mps 双向 backfill (D-5)
# v1 → v2 向后兼容: 调用方传 V0/V1/V2/V3 各档位的 drift_bias_sigma_mps 仍可用,
# continuous window stable_window_start(selection_mode="window") 仍是默认行为.
CONFIG_VERSION = 1  # 配置版本（预留，待配置校验链路完善后启用）。
OUTPUT_CONTRACT_VERSION = 1  # 输出契约版本。
SNAPSHOT_VERSION = 1  # 快照版本。


def summarize_versions() -> dict[str, int]:
    """汇总所有冻结版本号。

    把本模块定义的四个版本号统一收集到一个字典中返回，
    方便下游一次性获取全部版本信息，用于日志记录、审计输出或 smoke check。

    注意：返回 dict 而非 MappingProxyType，因为下游 write_json 使用标准
    json.dump 序列化，不支持 MappingProxyType。调用者不应修改返回值。

    返回：
        dict[str, int]：包含以下键的版本摘要字典：
        - "protocol_version"：整体协议版本号。
        - "config_version"：配置文件格式版本号。
        - "output_contract_version"：输出契约版本号。
        - "snapshot_version"：快照版本号。
    """
    return {  # 返回统一视图。调用者不应修改此字典。
        "protocol_version": PROTOCOL_VERSION,  # 协议版本。
        "config_version": CONFIG_VERSION,  # 配置版本。
        "output_contract_version": OUTPUT_CONTRACT_VERSION,  # 输出契约版本。
        "snapshot_version": SNAPSHOT_VERSION,  # 快照版本。
    }


from liquidloc.common.tee_logger import print_dict  # 局部导入参数打印工具。

print_dict(  # 打印本模块全部冻结版本号常量，供审计与 smoke check 核对。
    {
        "PROTOCOL_VERSION": PROTOCOL_VERSION,
        "CONFIG_VERSION": CONFIG_VERSION,
        "OUTPUT_CONTRACT_VERSION": OUTPUT_CONTRACT_VERSION,
        "SNAPSHOT_VERSION": SNAPSHOT_VERSION,
    },
    "version.py 常量",
    prefix="[配置]",
)
