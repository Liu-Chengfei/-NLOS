"""数据读写、适配器和数据集注册表的顶层导出入口。

这个包把准备脚本和管线最常用的数据层入口统一暴露出来，包含读取
原始数据、检查数据集就绪情况、映射字段名以及构造规范事件等能力。
这个文件只负责提供稳定的导入面，不负责承载具体逻辑，这样上层调用
时就不用记住内部子模块的文件结构。
"""

from liquidloc.dataio.adapters.event_builder import (  # 从包根重新导出事件构造函数，方便直接构造事件流。
    build_imu_events,  # IMU 事件构造函数。
    build_uwb_events,  # UWB 事件构造函数。
    build_vio_events,  # VIO 事件构造函数。
    build_flow_events,  # 光流/流量类事件构造函数（UTIL 数据集）。
    build_tof_events,  # ToF 测距类事件构造函数（UTIL 数据集）。
    merge_and_finalize_events,  # 事件合并与校验函数。
)
from liquidloc.dataio.adapters.field_mapper import map_external_fields  # 重新导出原始字段映射函数，供准备流程直接调用。
from liquidloc.dataio.readers.miluv_reader import (  # 从包根重新导出 MILUV 专用 reader 和就绪检查。
    audit_miluv_official_sample,  # 检查官方 MILUV 样例是否符合预期结构。
    inspect_miluv_raw_readiness,  # 检查 MILUV 原始目录是否已经可以处理。
    read_miluv_sequence,  # 读取一条 MILUV 序列并转成内部表示。
)
from liquidloc.dataio.readers.sim_reader import read_sim_sequence  # 重新导出仿真序列 reader。
from liquidloc.dataio.readers.util_reader import read_util_sequence  # 重新导出工具数据序列 reader。
from liquidloc.dataio.registry.public_dataset_registry import (  # 重新导出注册表读取函数，让数据集发现集中管理。
    get_dataset_entry,  # 按名称取出一个数据集条目。
    load_public_dataset_registry,  # 加载公开数据集注册表定义。
)

__all__ = (  # 显式声明 dataio 顶层公开 API，避免通配导入结果不稳定。
    "audit_miluv_official_sample",  # MILUV 样例审计函数。
    "build_imu_events",  # IMU 事件构造函数。
    "build_uwb_events",  # UWB 事件构造函数。
    "build_vio_events",  # VIO 事件构造函数。
    "build_flow_events",  # 光流/流量类事件构造函数。
    "build_tof_events",  # ToF 测距类事件构造函数。
    "get_dataset_entry",  # 注册表条目查询函数。
    "inspect_miluv_raw_readiness",  # MILUV 就绪检查函数。
    "load_public_dataset_registry",  # 注册表加载函数。
    "map_external_fields",  # 字段映射函数。
    "merge_and_finalize_events",  # 事件合并函数。
    "read_miluv_sequence",  # MILUV 读取函数。
    "read_sim_sequence",  # 仿真数据读取函数。
    "read_util_sequence",  # 工具数据读取函数。
)
