"""合同烟雾测试流水线。

这个流水线只做一件非常小的事: 构造一组最小可运行样本，
把输出目录、事件序列、预测文件、指标文件和协议快照都写出来，
用来检查整个工程的基础接口是否还能连通。

它不是正式实验流程，只是 scaffold 验证和最小闭环检查入口。

上游依赖:
- liquidloc.protocol.event_schema: 事件序列协议校验
- liquidloc.protocol.metric_schema: 指标顺序协议
- liquidloc.protocol.output_contract_schema: 输出合同校验
- liquidloc.protocol.version: 协议版本快照

下游调用者:
- CI/CD 验证脚本
- 开发者本地冒烟测试

核心变量:
- output_root: 烟雾测试输出目录
- events: 构造的最小事件序列
"""

from __future__ import annotations  # 允许在标注里使用当前模块相关类型。

import csv  # 用于写出最小的 CSV 指标文件。

from liquidloc.common.constants import DEFAULT_OUTPUT_DIRS  # 标准输出子目录列表。
from liquidloc.common.io_utils import write_json
from liquidloc.common.paths import resolve_output_root  # 统一的输出根目录解析函数。
from liquidloc.common.types import StageResult  # 流水线运行结果的统一返回结构。
from liquidloc.interfaces.pipeline_api import PipelineAPI, normalize_pipeline_cfg  # 流水线接口基类与统一配置规整 helper。
from liquidloc.protocol.event_schema import validate_event_sequence  # 校验事件序列是否符合协议。
from liquidloc.protocol.metric_schema import get_metric_order  # 读取指标顺序，保证输出稳定。
from liquidloc.protocol.output_contract_schema import check_output_contract  # 检查输出目录是否满足合同要求。
from liquidloc.protocol.version import summarize_versions  # 记录协议与版本快照。


class ContractSmokePipeline(PipelineAPI):
    """最小合同烟雾测试流水线。

    构造最小可运行样本，写出预测文件、指标文件和协议快照，
    检查整个工程的基础接口是否还能连通。
    """

    def run(self, pipeline_cfg: dict | None = None, runtime_context: dict | None = None) -> StageResult:
        """执行最小闭环检查。

        参数:
        - pipeline_cfg: 运行配置，主要控制项目根目录和输出目录。
        - runtime_context: 运行时上下文，这里保留接口但不使用。

        返回:
        - StageResult，其中包含产物路径和协议报告。
        """
        from liquidloc.common.tee_logger import print_dict
        print_dict({
            "pipeline_cfg": pipeline_cfg,
            "runtime_context_keys": list(runtime_context.keys()) if isinstance(runtime_context, dict) else None,
        }, "ContractSmokePipeline.run")
        cfg = normalize_pipeline_cfg(pipeline_cfg)  # 没传配置就用空字典，非映射配置直接拒绝。
        output_root = resolve_output_root(cfg, "mini_smoke")  # 统一解析输出根目录（含空白校验、Path(".") 拒绝、expanduser、类型检查、resolve）。
        for dirname in DEFAULT_OUTPUT_DIRS:  # 先把标准输出子目录都建好。
            (output_root / dirname).mkdir(parents=True, exist_ok=True)

        events = [  # 构造一组最小事件序列，覆盖 imu 和 uwb 两种模态。
            {
                "t": 0.0,  # 第一条事件时间戳。
                "dt": 0.0,  # 第一条事件的时间间隔固定为 0。
                "modality": "imu",  # 第一条是 IMU 事件。
                "meta": {"scene_id": "S(A0,N0,V0,K1)", "seq_id": "toy_seq"},  # 场景和序列标识。
                "imu_payload": {"ax": 0.0, "ay": 0.0, "gz": 0.0},  # 最小 IMU payload。
                "uwb_payload": None,  # 这一条没有 UWB 数据。
                "vio_payload": None,  # 这一条没有 VIO 数据。
            },
            {
                "t": 0.1,  # 第二条事件时间戳。
                "dt": 0.1,  # 与前一条的时间差。
                "modality": "uwb",  # 第二条是 UWB 事件。
                "meta": {"scene_id": "S(A0,N0,V0,K1)", "seq_id": "toy_seq"},  # 仍然使用同一场景和序列。
                "imu_payload": None,  # 这一条没有 IMU 数据。
                "uwb_payload": {"anchor_id": 0, "range": 2.0, "valid": True, "quality": 0.95},  # 最小 UWB payload。
                "vio_payload": None,  # 这一条没有 VIO 数据。
            },
        ]
        validate_event_sequence(events)  # 先确认事件结构符合协议。

        prediction_path = output_root / "predictions" / "mini_predictions.json"  # 预测文件输出位置。
        write_json(
            prediction_path,
            {
                "seq_id": "toy_seq",  # 这份预测对应的序列 ID。
                "scene_id": "S(A0,N0,V0,K1)",  # 这份预测对应的场景 ID。
                "timestamps": [event["t"] for event in events],  # 直接把事件时间戳写进预测文件。
                "states": [{"px": 0.0, "py": 0.0}, {"px": 0.1, "py": 0.0}],  # 最小状态序列示例。
            },
        )

        metrics_path = output_root / "metrics" / "mini_metrics.csv"  # 指标文件输出位置。
        with metrics_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["metric", "value"])  # 指标表头只保留 metric 和 value。
            writer.writeheader()  # 先写表头。
            for metric in get_metric_order()[:3]:  # 只写前三个指标，保持烟雾测试足够小。
                writer.writerow({"metric": metric, "value": 0.0})  # 每个指标都写一个占位值。

        snapshot_path = output_root / "audits" / "protocol_snapshot.json"  # 协议快照位置。
        write_json(snapshot_path, summarize_versions())  # 把版本摘要写出来。

        report = check_output_contract(output_root)  # 检查输出目录是否满足合同要求。
        log_path = output_root / "logs" / "mini_smoke.log"  # 日志文件位置。
        write_json(log_path, report)  # 把合同报告也写成日志。

        return StageResult(
            stage_name="contract_smoke",  # 当前阶段名。
            artifacts=[str(prediction_path), str(metrics_path), str(snapshot_path), str(log_path)],  # 所有产物路径。
            metadata={"contract_report": report},  # 元数据里记录合同检查报告。
        )
