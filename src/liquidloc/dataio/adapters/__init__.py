"""把公开数据集的原始表转换成内部事件表的适配器。

这个包就是数据集原始记录和内部稳定事件结构之间的窄桥。这里导出的
内容故意保持很少，因为这些函数会被 readers、manifests 以及更高层的
准备流程反复调用。这个文件本身不承载业务逻辑，只负责把适配器入口
集中到一个固定位置，方便上层统一导入，也方便你检查对外暴露了什么。
"""

from liquidloc.dataio.adapters.event_builder import (  # 把事件构造函数从专门实现文件重新导出，调用方就不用关心内部拆分。
    build_imu_events,  # 把归一化后的原始行构造成 IMU 事件字典。
    build_uwb_events,  # 把归一化后的原始行构造成 UWB 事件字典。
    build_vio_events,  # 把归一化后的原始行构造成 VIO 事件字典。
    build_flow_events,  # 把归一化后的原始行构造成光流/流量类事件字典（UTIL 数据集）。
    build_tof_events,  # 把归一化后的原始行构造成 ToF 测距类事件字典（UTIL 数据集）。
    merge_and_finalize_events,  # 合并多模态事件流并写入最终 dt。
)
from liquidloc.dataio.adapters.field_mapper import map_external_fields  # 重新导出外部字段映射函数，方便上层直接调用。

__all__ = (  # 明确声明这个包对外公开的适配器入口，避免通配导入时出现不稳定行为。
    "build_imu_events",  # IMU 事件构造入口。
    "build_uwb_events",  # UWB 事件构造入口。
    "build_vio_events",  # VIO 事件构造入口。
    "build_flow_events",  # 光流/流量类事件构造入口。
    "build_tof_events",  # ToF 测距类事件构造入口。
    "map_external_fields",  # 字段映射入口。
    "merge_and_finalize_events",  # 事件合并与收尾入口。
)
