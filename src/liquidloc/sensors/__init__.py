"""传感器测量提取与规范化对外入口。

本模块是 liquidloc.sensors 包的统一导出层，把各子模块的公开 API 汇聚到一起。
注意：经仓内核查，当前所有下游消费方均通过 ``from liquidloc.sensors.<submodule> import ...``
直连子模块使用，无任何代码通过 ``from liquidloc.sensors import ...`` 走门面。
本门面仅作为模块导出声明（``__all__``）保留，便于检索公开 API 范围，不作为推荐调用入口；
新增消费方应延续现有惯例，直连对应子模块。

上游依赖：
    liquidloc.sensors.imu_model（extract_imu_measurement, validate_imu_event）
    liquidloc.sensors.uwb_model（extract_uwb_measurement, predict_range_to_anchor）
    liquidloc.sensors.vision_model（extract_vio_measurement）
    liquidloc.sensors.anchor_model（build_anchor_lookup, compute_geometry_report, validate_anchor_layout, project_anchor_layout_xy）
    liquidloc.sensors.quality_model（get_event_quality, normalize_quality_value）

下游调用者（共 9 个，全部直连子模块，不走门面）：
    liquidloc.estimators.predict_step（extract_imu_measurement，直连 imu_model）
    liquidloc.estimators.vision_update_step（extract_vio_measurement，直连 vision_model）
    liquidloc.estimators.ekf_core（build_anchor_lookup，直连 anchor_model）
    liquidloc.dataio.sim_materializer（predict_range_to_anchor, build_anchor_lookup，直连 uwb_model/anchor_model）
    liquidloc.pipelines.core_pipeline（predict_range_to_anchor, build_anchor_lookup，直连 uwb_model/anchor_model）
    liquidloc.pipelines.train_pipeline（extract_uwb_measurement, predict_range_to_anchor, build_anchor_lookup, project_anchor_layout_xy，直连 uwb_model/anchor_model）
    liquidloc.pipelines.miluv_pipeline（project_anchor_layout_xy，直连 anchor_model）
    liquidloc.models.features.feature_builder（predict_range_to_anchor, compute_geometry_report，直连 uwb_model/anchor_model）
    liquidloc.scenarios.geometry_levels（validate_anchor_layout, _compute_geometry_score_core，直连 anchor_model）

核心导出变量：
    __all__: 包级公开 API 名称元组，控制 ``from liquidloc.sensors import *`` 的导出范围。
"""

from liquidloc.sensors.anchor_model import (  # 锚点几何相关工具统一对外导出。
    build_anchor_lookup,  # 锚点 id 到坐标的查表函数。
    compute_geometry_report,  # 锚点几何评分报告函数。
    project_anchor_layout_xy,  # 3D 锚点布局投影到 2D 的函数。
    validate_anchor_layout,  # 锚点布局校验函数。
)
from liquidloc.sensors.imu_model import extract_imu_measurement, validate_imu_event  # IMU 事件读取和校验统一对外导出。
from liquidloc.sensors.quality_model import get_event_quality, normalize_quality_value  # 质量值读取和归一化统一对外导出。
from liquidloc.sensors.uwb_model import extract_uwb_measurement, predict_range_to_anchor  # UWB 测量提取和距离预测统一对外导出。
from liquidloc.sensors.vision_model import extract_vio_measurement  # VIO 测量提取统一对外导出。

__all__ = (  # 明确包级公开 API。
    "build_anchor_lookup",  # 锚点查表函数名。
    "compute_anchor_gdop_report",  # 锚点 GDOP 报告（准则 §4 合规验证）。
    "compute_geometry_report",  # 锚点几何报告函数名。
    "extract_imu_measurement",  # IMU 测量提取函数名。
    "extract_uwb_measurement",  # UWB 测量提取函数名。
    "extract_vio_measurement",  # VIO 测量提取函数名。
    "get_event_quality",  # 事件质量读取函数名。
    "normalize_quality_value",  # 质量值归一化函数名。
    "predict_range_to_anchor",  # anchor 距离预测函数名。
    "project_anchor_layout_xy",  # 3D→2D 锚点布局投影函数名。
    "validate_anchor_layout",  # 锚点布局校验函数名。
    "validate_imu_event",  # IMU 事件校验函数名。
)
