"""数据集清单构建、契约检查和切分规则的公开导出层。

这个包提供数据集准备流程中三个核心能力：
1. **契约检查**（dataset_checks）：校验原始包和序列目录是否满足最低文件
   和时间戳契约，以及布局族覆盖和 SIM 物化基准合同。
2. **清单构建**（build_manifests）：扫描数据根目录，产出按序列和按场景
   组织的两份清单。
3. **切分构建**（split_builder）：基于清单构建 train/val/test 切分，并
   检查跨切分泄漏（重复序列、共享作用域、共享布局族）。

本文件只负责把这三个子模块最常用的入口集中导出给上层调用者，避免
调用方直接记住内部文件路径。这样做的目的是让 dataio.manifests 成为
一个稳定的门面，上游只要导入这个包，就能拿到清单和切分相关的核心工具。

上游依赖：
- dataio.manifests.dataset_checks（契约常量和检查函数）
- dataio.manifests.build_manifests（清单构建函数和流文件常量）
- dataio.manifests.split_builder（切分构建和泄漏检查函数）

下游调用者：
- 准备流程脚本（构建清单、切分和检查契约）
- 公开验证流程（确认数据集满足最低契约和切分公平性）
"""

from liquidloc.dataio.manifests.build_manifests import REQUIRED_STREAMS, build_manifests
from liquidloc.dataio.manifests.dataset_checks import (
    ANCHOR_LAYOUT_FILENAME,
    REQUIRED_RAW_KEYS,
    REQUIRED_SEQ_FILES,
    SIM_MATERIALIZED_BASELINE_ANCHOR_COUNT,
    SIM_MATERIALIZED_BASELINE_K_LEVEL,
    inspect_layout_family_coverage,
    inspect_sim_materialized_contract,
    run_dataset_checks,
)
from liquidloc.dataio.manifests.split_builder import build_splits

__all__ = (
    "REQUIRED_RAW_KEYS",
    "REQUIRED_SEQ_FILES",
    "ANCHOR_LAYOUT_FILENAME",
    "SIM_MATERIALIZED_BASELINE_ANCHOR_COUNT",
    "SIM_MATERIALIZED_BASELINE_K_LEVEL",
    "REQUIRED_STREAMS",
    "build_manifests",
    "build_splits",
    "inspect_layout_family_coverage",
    "inspect_sim_materialized_contract",
    "run_dataset_checks",
)
