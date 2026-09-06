"""公开导出各类原始数据读取入口，方便上层统一导入。"""

from pathlib import Path  # 统一处理文件路径。

from liquidloc.dataio.readers.miluv_reader import (  # 重新导出 MILUV 读取与就绪检查函数。
    audit_miluv_official_sample,  # 审计官方 MILUV 样例元数据是否满足本地约定。
    inspect_miluv_raw_readiness,  # 检查 MILUV 原始目录是否可用于端到端烟雾测试。
    read_miluv_sequence,  # 读取一条 MILUV 序列并返回原始包。
)
from liquidloc.dataio.readers.ntu_viral_reader import (  # 重新导出 NTU VIRAL 读取与就绪检查函数。
    inspect_ntu_viral_raw_readiness,  # 检查 NTU VIRAL 原始目录是否可用于公开基准烟雾测试。
    read_ntu_viral_sequence,  # 读取一条 NTU VIRAL 序列并返回原始包。
)
from liquidloc.dataio.readers.sim_reader import read_sim_sequence  # 重新导出仿真数据读取入口。
from liquidloc.dataio.readers.util_reader import (  # 重新导出 UTIL 读取与就绪检查函数。
    inspect_util_raw_readiness,  # 检查 UTIL 原始目录是否可用于烟雾测试。
    read_util_sequence,  # 读取一条 UTIL 序列并返回原始包。
)


def list_sequence_dirs(raw_root: Path) -> list[Path]:
    """列出数据根目录下的有效序列目录，排除 config 和隐藏目录。

    统一过滤规则：只保留 is_dir() 且名称不为 'config' 且不以 '.' 开头的子目录，
    按目录名排序返回。根目录不存在时返回空列表。

    参数：
        raw_root: 数据根目录路径。

    返回：
        排序后的有效序列目录列表。
    """
    if not raw_root.is_dir():  # 根目录不存在时返回空列表。
        return []
    return sorted(  # 排序后返回，确保输出稳定。
        [path for path in raw_root.iterdir() if path.is_dir() and path.name != 'config' and not path.name.startswith('.')],
        key=lambda path: path.name,  # 按目录名排序。
    )


__all__ = (  # 显式声明本包对外公开的读取 API。
    "audit_miluv_official_sample",  # MILUV 官方样例审计入口。
    "inspect_miluv_raw_readiness",  # MILUV 原始目录就绪检查入口。
    "inspect_ntu_viral_raw_readiness",  # NTU VIRAL 原始目录就绪检查入口。
    "list_sequence_dirs",  # 序列目录列表函数。
    "read_miluv_sequence",  # MILUV 序列读取入口。
    "read_ntu_viral_sequence",  # NTU VIRAL 序列读取入口。
    "read_sim_sequence",  # 仿真序列读取入口。
    "inspect_util_raw_readiness",  # UTIL 原始目录就绪检查入口。
    "read_util_sequence",  # 工具序列读取入口。
)
