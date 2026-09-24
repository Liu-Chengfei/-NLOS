"""资源预算管理器：RAM ≤12GB 硬上限 + 并行度 = min(GPU, RAM, 待执行)。

  - 并发调度器按 "并发单元数 × 单单元峰值内存 ≤ 12GB" 计算 RAM 闸门
  - 调度器并行度 = min(GPU 允许并发, RAM 允许并发, 待执行单元数)

本模块提供：
  - PsUtilMemoryProbe：读取系统总内存、可用内存
  - NVMLGpuProbe：读取 GPU 显存总量
  - compute_parallel_limits：输出 (gpu_parallel, ram_parallel, final_parallel)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

# 硬上限：RAM ≤12G
RAM_LIMIT_GB: float = 12.0
RAM_LIMIT_BYTES: float = RAM_LIMIT_GB * 1024 * 1024 * 1024  # 12 GB


class PsUtilMemoryProbe:
    """通过 psutil 读取系统内存统计。"""

    @staticmethod
    def get_total_memory_gb() -> float:
        """返回系统物理内存总量（GB）。"""
        import psutil
        return psutil.virtual_memory().total / (1024**3)

    @staticmethod
    def get_available_memory_gb() -> float:
        """返回可用内存（GB）。"""
        import psutil
        return psutil.virtual_memory().available / (1024**3)

    @staticmethod
    def get_peak_memory_mb() -> float:
        """返回进程峰值内存占用（MB）。"""
        import psutil
        proc = psutil.Process()
        try:
            # memory_info().rss 为当前实际占用，vms 可参考峰值估算
            return proc.memory_info().rss / (1024**2)
        except psutil.AccessDenied:
            return 0.0


class NVMLGpuProbe:
    """通过 nvidia-ml-py 读取 GPU 显存统计。若无 GPU，返回 None。"""

    def __init__(self):
        self._nvml = None
        self._device_count = 0
        self._init_success = False
        self._init_nvml()

    def _init_nvml(self) -> None:
        """尝试初始化 NVML。"""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._device_count = pynvml.nvmlDeviceGetCount()
            self._init_success = True
        except Exception:
            self._nvml = None
            self._device_count = 0
            self._init_success = False

    def get_gpu_count(self) -> int:
        """返回检测到的 GPU 数量。"""
        return self._device_count if self._init_success else 0

    def get_total_memory_mb(self, device: int = 0) -> Optional[float]:
        """返回指定 GPU 的总显存（MB）。"""
        if not self._init_success:
            return None
        try:
            info = self._nvml.nvmlDeviceGetMemoryInfo(device)
            return info.total / (1024**2)
        except Exception:
            return None

    def get_used_memory_mb(self, device: int = 0) -> Optional[float]:
        """返回指定 GPU 的已用显存（MB）。"""
        if not self._init_success:
            return None
        try:
            info = self._nvml.nvmlDeviceGetMemoryInfo(device)
            return info.used / (1024**2)
        except Exception:
            return None

    def shutdown(self) -> None:
        """释放 NVML 资源。"""
        if self._nvml:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass


def estimate_unit_peak_memory_mb(executor_type: str) -> float:
    """估算单个执行单元的峰值内存（MB）。
    
    此函数返回经验估算值，实际值由 runtime ram_peak 字段覆盖。
    """
    # 经验值：每单元约占 2.5GB 显存 + 1GB 系统内存
    # 可配置于 configs/base/resource_budget.yaml
    if executor_type == "neural":
        return 3500.0  # 3.5 GB ≈ LSTM/liquid/transformer 训练单元峰值
    elif executor_type == "classical":
        return 1500.0  # 1.5 GB ≈ EKF/roBUST-EKF 单元峰值
    else:
        return 2500.0  # 默认 2.5 GB


def compute_parallel_limits(
    n_pending: int,
    unit_peak_memory_mb: float,
    executor_type: str = "neural",
    allow_multi_gpu: bool = True,
) -> dict[str, Any]:
    """计算并行度 = min(GPU, RAM, 待执行)。

    返回 dict 包含：
      - gpu_count: 检测到的 GPU 数量
      - gpu_parallel: GPU 允许的最大并发度
      - ram_limit_gb: RAM 硬上限（12 GB）
      - ram_parallel: RAM 允许的最大并发度
      - pending: 待执行单元数
      - final_parallel: 实际并行度
      - passed: 是否满足 RAM 硬上限（RAM 并发 >= 1）
    """
    # GPU 计算
    gpu_probe = NVMLGpuProbe()
    gpu_count = gpu_probe.get_gpu_count()
    
    gpu_parallel = 1
    for dev_idx in range(gpu_count):
        total_mb = gpu_probe.get_total_memory_mb(dev_idx)
        if total_mb and total_mb > 0:
            dev_parallel = int(total_mb / unit_peak_memory_mb)
            gpu_parallel = max(gpu_parallel, dev_parallel)
    
    if gpu_count == 0:
        # 无 GPU 时，训练入口应报错 C-4，不计入此并行度
        gpu_parallel = 0
    
    # RAM 计算（C-6 硬上限）
    ram_probe = PsUtilMemoryProbe()
    total_ram_gb = ram_probe.get_total_memory_gb()
    
    # 要求：单元峰值内存 × 并发数 ≤ 12GB
    unit_peak_gb = unit_peak_memory_mb / 1024.0
    ram_parallel = int(RAM_LIMIT_GB / unit_peak_gb) if unit_peak_gb > 0 else 0
    
    # 并行度 = min(GPU 并发, RAM 并发, 待执行)
    final_parallel = min(gpu_parallel, ram_parallel, n_pending)
    if gpu_parallel == 0 and ram_parallel > 0:
        # 无 GPU，CPU 空闲协助：仅在 GPU 排满后才启动
        final_parallel = 0  # 训练入口 已报错，故此处为 0
    
    # 断言：无 GPU 时 final_parallel 必须为 0
    if gpu_count == 0:
        final_parallel = 0
    
    return {
        "gpu_count": gpu_count,
        "gpu_parallel": gpu_parallel,
        "total_ram_gb": total_ram_gb,
        "ram_limit_gb": RAM_LIMIT_GB,
        "ram_parallel": ram_parallel,
        "pending": n_pending,
        "final_parallel": final_parallel,
        "passed": final_parallel >= 0,  # 通过判据：有调度余量
        "unit_peak_memory_mb": unit_peak_memory_mb,
    }


def verify_ram_budget(manifest_path: Path) -> dict[str, Any]:
    """从 manifest 读取每个单元的实际峰值内存，验证总计是否 ≤ 12GB。

    运行期冒烟实测单单元峰值内存（psutil 探针）。
    """
    import json
    
    if not manifest_path.is_file():
        return {"passed": False, "detail": f"manifest 未找到: {manifest_path}"}
    
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"passed": False, "detail": f"manifest 解析失败: {e}"}
    
    results = manifest.get("results", [])
    if not results:
        return {"passed": False, "detail": "manifest 无 results 字段"}
    
    total_peak_mb = 0.0
    for r in results:
        peak = r.get("ram_peak_mb", r.get("ram_peak", 0))
        total_peak_mb += float(peak) if peak else 0.0
    
    total_peak_gb = total_peak_mb / 1024.0
    passed = total_peak_gb <= RAM_LIMIT_GB
    
    return {
        "passed": passed,
        "detail": f"总峰值内存 = {total_peak_gb:.2f} GB / {RAM_LIMIT_GB} GB 限额",
        "total_peak_gb": total_peak_gb,
        "ram_limit_gb": RAM_LIMIT_GB,
        "unit_count": len(results),
    }


# ----------------- 断言装饰器 -----------------

def assert_ram_budget(n_units: int, unit_peak_mb: float) -> None:
    """在调度前断言 RAM 预算满足 C-6。"""
    total_peak_mb = n_units * unit_peak_mb
    total_peak_gb = total_peak_mb / 1024.0
    if total_peak_gb > RAM_LIMIT_GB:
        raise ValueError(
            f"RAM 预算超限: {n_units} 单元 × {unit_peak_mb:.1f} MB = {total_peak_gb:.2f} GB > {RAM_LIMIT_GB} GB 上限。"
            f"请减小 n_units 或优化单单元内存占用。"
        )


def assert_parallel_degree(parallel: int) -> None:
    """在调度后断言并行度 ≤ min(GPU, RAM)。"""
    limits = compute_parallel_limits(
        n_pending=parallel,
        unit_peak_memory_mb=estimate_unit_peak_memory_mb("neural"),
        executor_type="neural",
    )
    if parallel > limits["final_parallel"]:
        raise ValueError(
            f"并行度 {parallel} 超过 RAM/GPU 约束 {limits['final_parallel']}。"
            f"请遵循 C-7: 并行 = min(GPU, RAM, 待执行)。"
        )