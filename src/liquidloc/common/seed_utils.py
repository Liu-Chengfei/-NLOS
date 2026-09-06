"""全局随机种子工具。

职责：
    把 Python、NumPy 和 PyTorch 的随机种子设置集中起来，并返回
    可审计的报告，方便训练、评估和 smoke test 复现。

上游依赖：
    - Python 标准库 random           — Python 随机源
    - NumPy（可选）                  — 数据采样与增强的随机源
    - PyTorch（可选）                — 训练与推理的随机源

下游调用者：
    - liquidloc.pipelines.train      — 训练流水线在启动前设置全局种子
    - liquidloc.pipelines.eval       — 评估流水线在启动前设置全局种子
    - scripts / smoke tests          — 脚本在入口处调用 set_global_seed 保证可复现

核心变量：
    - _numpy_rng                    — 模块级 NumPy Generator 实例（set_global_seed 时初始化）
"""

from __future__ import annotations

import os  # 环境变量操作。
import random  # Python 标准随机源。

from liquidloc.common.validation import is_bool_like, is_integer

__all__ = (
    "cuda_runtime_usable",
    "set_global_seed",
    "build_seed_report",
    "get_numpy_rng",
)

_numpy_rng = None  # 模块级 NumPy Generator 实例，由 set_global_seed 初始化。


def cuda_runtime_usable() -> bool:  # 判断当前 PyTorch 运行时是否真的能执行 CUDA。
    """判断当前 PyTorch 运行时是否真的能执行 CUDA 运算。

    依次检查：torch 是否可导入 → CUDA API 是否声明可用 → 底层 CUDA 设备
    计数接口是否存在 → 能否完成一次最小的 CUDA 张量操作。只有全部通过
    才返回 True。

    此函数比 `torch.cuda.is_available()` 更严格，因为某些环境可能存在
    CUDA API 但无法实际执行 CUDA 操作（如驱动版本不匹配、GPU 被占用等）。

    Args:
        无参数。

    Returns:
        bool: True 表示 CUDA 运行时真正可用，False 表示不可用。

    设计意图
    ---------
    本函数位于 common 层但使用 CUDA 张量，这是有意为之的检测逻辑而非计算逻辑：
    - 目的是验证 CUDA 运行时是否真正可用，而非执行 GPU 计算
    - CUDA 张量操作被 try/except 包裹，无 CUDA 环境时安全返回 False
    - 本函数不违反 AGENTS.md 第 5 节"CPU 默认"规则，因为它只做可用性检测
    - 调用方应根据返回值决定是否使用 GPU，而非假设 GPU 可用
    """
    try:  # 先尝试导入 torch。
        import torch  # 只在函数内部导入，避免没有 torch 时影响别的工具。
    except ImportError:  # 没有 torch 就肯定不可用。
        return False  # 直接返回不可用。

    if not torch.cuda.is_available():  # CUDA API 自己都说不可用，就不用继续测了。
        return False  # 返回不可用。

    if not hasattr(torch._C, "_cuda_getDeviceCount"):  # 某些环境虽然有接口，但底层 CUDA 能力不完整。
        return False  # 视为不可用。

    try:  # 再做一次真实 CUDA 张量操作，避免只看 API 就误判。
        # 这里做一次最小的真实 CUDA 张量操作，避免只看 API 存在就误判可用。
        torch.empty(1, device="cuda").cpu()  # 创建一个 CUDA 张量再搬回 CPU。
    except Exception:  # 任何异常都说明 CUDA 运行时不可用（含 OSError 等驱动层异常）。
        return False  # 返回不可用。
    return True  # 能完成最小 CUDA 操作才算真的可用。


def set_global_seed(seed: int, deterministic: bool = True) -> dict:  # 设置全局随机种子并返回审计报告。
    """设置全局随机种子，并返回可审计的种子报告。

    依次设置 Python 标准库 random、NumPy（如可用）和 PyTorch（如可用）
    的随机种子。当 deterministic=True 时，还会开启 cuDNN 确定性模式和
    PyTorch 全局确定性算法开关。

    Args:
        seed (int): 随机种子值，必须是非负整数且小于 2**32。
        deterministic (bool): 是否开启确定性模式，默认为 True。
            开启后牺牲少量性能换取更好的可复现性。

    Returns:
        dict: 种子设置审计报告，包含以下键：
            - seed (int): 本次设置的种子值。
            - deterministic (bool): 是否开启确定性模式。
            - python_seeded (bool): Python 标准库是否已设置（始终为 True）。
            - numpy_seeded (bool): NumPy 是否已设置。
            - torch_seeded (bool): PyTorch 是否已设置。
            - cuda_seeded (bool): CUDA 种子是否已设置。
            - cudnn_deterministic_set (bool): cuDNN 确定性是否已设置。
            - deterministic_algorithms_set (bool): 全局确定性算法是否已设置。

    Raises:
        TypeError: 当 seed 不是整数或是 bool，或 deterministic 不是 bool 时抛出。
        ValueError: 当 seed 为负数或大于等于 2**32 时抛出。

    Note:
        本函数不设置 ``PYTHONHASHSEED`` 环境变量。Python 的 hash 随机化
        在进程启动时确定，运行时无法修改。如需完全确定性，请在启动脚本
        中设置 ``PYTHONHASHSEED=0``。

        当 ``deterministic=True`` 且 CUDA 可用时，本函数会设置
        ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` 环境变量（仅当该变量未
        已设置时），这是 cuDNN 某些操作实现确定性所必需的。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"seed": seed, "deterministic": deterministic}, "set_global_seed 入口参数")
    if not is_integer(seed):  # seed 必须是整数，不能是 bool。
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}")  # 类型不对就报错。
    seed = int(seed)  # 统一转为 Python int，避免 numpy.integer 泄漏到返回值。
    if seed < 0:  # 种子不能是负数。
        raise ValueError(f"seed must be non-negative, got {seed}")  # 负数不允许。
    if seed >= 2**32:  # 为了兼容常见随机库，种子上限要受控。
        raise ValueError(f"seed must be less than 2**32, got {seed}")  # 超出范围就报错。
    if not is_bool_like(deterministic):  # deterministic 必须是布尔值。
        raise TypeError(f"deterministic must be a boolean, got {type(deterministic).__name__}")  # 类型不对就报错。
    deterministic = bool(deterministic)  # 统一转为 Python bool，避免 numpy.bool_ 泄漏到返回值和 JSON 序列化。

    random.seed(seed)  # 先设置 Python 标准库的随机种子。

    numpy_seeded = False  # 先假设 NumPy 没有可用。
    torch_seeded = False  # 先假设 PyTorch 没有可用。
    cuda_seeded = False  # 记录 CUDA 种子是否设置。
    cudnn_deterministic_set = False  # 记录 cuDNN 确定性是否设置。
    deterministic_algorithms_set = False  # 记录全局确定性算法是否设置。

    try:  # 尝试导入 numpy。
        import numpy as np  # NumPy 有就顺手同步设置。
    except ImportError:  # 没有 NumPy 就把标记设为空。
        np = None  # 用 None 表示不可用。

    if np is not None:  # NumPy 可用时同步设置。
        # NumPy 2.0+ 兼容：使用 SeedSequence + Generator 替代已废弃的 np.random.seed。
        _ss = np.random.SeedSequence(seed)  # 用 SeedSequence 派生可复现的种子序列。
        global _numpy_rng  # 写入模块级 Generator 实例。
        _numpy_rng = np.random.default_rng(_ss)  # 创建 Generator 实例。
        numpy_seeded = True  # 记录 NumPy 已设置。

    try:  # 再尝试导入 torch。
        import torch  # PyTorch 可用时也要同步设置。
    except ImportError:  # 没有 torch 就不做 PyTorch 设置。
        torch = None  # 用 None 表示不可用。

    if torch is not None:  # PyTorch 可用时进行同步设置。
        # PyTorch 主随机源。
        torch.manual_seed(seed)  # 设置 CPU 侧随机种子。
        if cuda_runtime_usable():  # 如果 CUDA 真的可用，就同步设置 GPU 侧种子。
            # CUDA 场景下也同步设置，避免 CPU/GPU 端出现不同步随机流。
            torch.cuda.manual_seed(seed)  # 设置当前 GPU 随机种子。
            torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 随机种子。
            cuda_seeded = True  # 记录 CUDA 种子已设置。
            # CUBLAS_WORKSPACE_CONFIG 是 CUDA 确定性所必需的环境变量。
            # 当 deterministic=True 时必须设置，否则 cuDNN 某些操作仍可能非确定性。
            if deterministic:
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        if hasattr(torch.backends, "cudnn"):  # 如果当前 torch 带 cuDNN 后端。
            # deterministic=True 时牺牲一点性能，换更稳定的复现结果。
            torch.backends.cudnn.deterministic = deterministic  # 开关 cuDNN 确定性模式。
            torch.backends.cudnn.benchmark = not deterministic  # 非确定性时允许 benchmark 优化。
            cudnn_deterministic_set = True  # 记录 cuDNN 确定性已设置。
        if hasattr(torch, "use_deterministic_algorithms"):  # 较新的 torch 支持全局确定性算法开关。
            torch.use_deterministic_algorithms(deterministic)  # 按需启用确定性算法。
            deterministic_algorithms_set = True  # 记录全局确定性算法已设置。

        # P19 硬约束：TF32 开关显式设置并写入 manifest（手册 line 211）
        # 三网络与 EKF/Robust-EKF 共用同一设置（不允许 LNN FP32 + LSTM TF32 等精度不一致）
        # 显式 False 保证跨机/跨版本浮点可复现（PyTorch 默认 Ampere+ 上 allow_tf32=True）
        _tf32_flag = bool(deterministic)
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = _tf32_flag
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = _tf32_flag
        _tf32_explicit_set = True

        torch_seeded = True  # 记录 PyTorch 已设置。

    return build_seed_report(  # 最后构造一份可审计的报告。
        seed=seed,  # 记录本次种子值。
        deterministic=deterministic,  # 记录是否开启确定性。
        numpy_seeded=numpy_seeded,  # 记录 NumPy 是否设置成功。
        torch_seeded=torch_seeded,  # 记录 PyTorch 是否设置成功。
        cuda_seeded=cuda_seeded,  # 记录 CUDA 种子是否设置。
        cudnn_deterministic_set=cudnn_deterministic_set,  # 记录 cuDNN 确定性是否设置。
        deterministic_algorithms_set=deterministic_algorithms_set,  # 记录全局确定性算法已设置。
        tf32_explicit_set=locals().get("_tf32_explicit_set", False),  # 记录 TF32 显式设置。
        tf32_enabled=_tf32_flag if locals().get("_tf32_explicit_set", False) else None,  # 记录 TF32 实际开关。
    )  # 报告构造结束。


def build_seed_report(  # 把种子设置结果整理成字典。
    seed: int,  # 当前种子值。
    *,  # 后面的参数必须关键字传入，便于未来扩展新框架种子状态而不破坏现有调用方。
    deterministic: bool,  # 是否启用确定性模式。
    numpy_seeded: bool,  # NumPy 是否设置成功。
    torch_seeded: bool,  # PyTorch 是否设置成功。
    cuda_seeded: bool = False,  # CUDA 种子是否设置。
    cudnn_deterministic_set: bool = False,  # cuDNN 确定性是否设置。
    deterministic_algorithms_set: bool = False,  # 全局确定性算法是否设置。
    tf32_explicit_set: bool = False,  # P19：TF32 显式设置标志。
    tf32_enabled: bool | None = None,  # P19：TF32 实际开关（None=未显式设置）。
) -> dict:  # 返回一个便于审计的字典。
    """构造当前种子设置的审计报告。

    Args:
        seed (int): 当前设置的种子值。
        deterministic (bool): 是否启用了确定性模式。
        numpy_seeded (bool): NumPy 随机种子是否设置成功。
        torch_seeded (bool): PyTorch 随机种子是否设置成功。
        cuda_seeded (bool): CUDA 随机种子是否设置成功，默认为 False。
        cudnn_deterministic_set (bool): cuDNN 确定性模式是否设置，默认为 False。
        deterministic_algorithms_set (bool): PyTorch 全局确定性算法是否设置，默认为 False。

    Returns:
        dict: 审计报告字典，包含以下键：
            - seed (int): 种子值。
            - deterministic (bool): 确定性开关状态。
            - python_seeded (bool): Python 标准库种子状态（始终 True）。
            - numpy_seeded (bool): NumPy 种子状态。
            - torch_seeded (bool): PyTorch 种子状态。
            - cuda_seeded (bool): CUDA 种子状态。
            - cudnn_deterministic_set (bool): cuDNN 确定性状态。
            - deterministic_algorithms_set (bool): 全局确定性算法状态。
    """
    return {  # 把审计信息集中返回。
        "seed": seed,  # 记录种子。
        "deterministic": deterministic,  # 记录确定性开关。
        "python_seeded": True,  # Python 标准库总是已经设置。
        "numpy_seeded": numpy_seeded,  # NumPy 设置状态。
        "torch_seeded": torch_seeded,  # PyTorch 设置状态。
        "cuda_seeded": cuda_seeded,  # CUDA 种子设置状态。
        "cudnn_deterministic_set": cudnn_deterministic_set,  # cuDNN 确定性设置状态。
        "deterministic_algorithms_set": deterministic_algorithms_set,  # 全局确定性算法设置状态。
        "tf32_explicit_set": tf32_explicit_set,  # P19：TF32 显式设置状态。
        "tf32_enabled": tf32_enabled,  # P19：TF32 实际开关（None=未显式）。
    }  # 报告字典结束。


def get_numpy_rng() -> numpy.random.Generator | None:  # 获取当前模块级 NumPy Generator 实例。
    """返回由 set_global_seed 初始化的 NumPy Generator 实例。

    下游代码应使用此函数获取 Generator，而非调用已废弃的 np.random.seed /
    np.random.* 全局函数。

    Args:
        无参数。

    Returns:
        numpy.random.Generator | None: Generator 实例；若 NumPy 不可用或
            尚未调用 set_global_seed，则返回 None。

    Note:
        返回类型标注为 ``numpy.random.Generator | None``，但运行时
        实际返回类型取决于 NumPy 是否已安装。当 NumPy 不可用时，
        此函数始终返回 None。

    Example:
        >>> set_global_seed(42)
        >>> rng = get_numpy_rng()
        >>> rng.random((3,))  # 使用 Generator 进行随机采样
        array([0.77395605, 0.43887844, 0.85859792])
    """
    return _numpy_rng
