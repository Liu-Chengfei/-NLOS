"""公开数据集注册表的对外导出层。

这个模块本身不承载业务逻辑，只负责把注册表读取、查询和名称枚举
这三个最常用的入口集中导出给上层调用者，避免调用方直接记住内部
文件路径。这样做的目的，是让 `dataio.registry` 成为一个稳定的门面，
上游只要导入这个包，就能拿到公开数据集注册相关的核心工具。
"""

from liquidloc.dataio.registry.public_dataset_registry import (  # 直接转出注册表查询能力，供上层统一导入。
    get_dataset_entry,  # 按数据集名称取出对应注册表条目。
    inspect_public_dataset_readiness,  # 检查单个数据集的公开验证就绪状态。
    load_public_dataset_registry,  # 读取公开数据集注册表 YAML。
    list_public_dataset_names,  # 枚举注册表里所有公开数据集名称。
)

__all__ = (
    "get_dataset_entry",  # 对外暴露"单条查询"入口。
    "inspect_public_dataset_readiness",  # 对外暴露"就绪状态检查"入口。
    "load_public_dataset_registry",  # 对外暴露"加载整张注册表"入口。
    "list_public_dataset_names",  # 对外暴露"名称列表"入口。
)
