"""日志初始化工具。

职责：
    统一处理 logger、formatter、console handler 和 file handler，
    避免每个脚本重复拼装日志配置。

上游依赖：
    - Python 标准库 logging          — 日志基础设施
    - liquidloc.common.paths         — 用 build_output_path 构造日志文件路径

下游调用者：
    - liquidloc.pipelines.*          — 流水线层在入口处创建 logger
    - liquidloc.protocol.*           — 协议层用 logger 记录校验与审计信息
    - scripts / notebooks            — 脚本在入口处创建 logger

核心变量：
    - 无模块级变量，全部通过函数返回
"""

from __future__ import annotations

import logging  # 用标准库 logging 搭建日志系统。
from pathlib import Path  # 用 Path 统一处理日志文件路径。

from liquidloc.common.paths import build_output_path  # 复用项目统一输出路径构造函数。
from liquidloc.common.validation import is_bool_like, is_integer as _is_integer, is_string_like  # 复用统一布尔/整数/字符串类型校验，防止 bool 泄漏为 int。

__all__ = ("get_logger",)


def get_logger(
    logger_name: str,
    *,
    log_path: str | Path | None = None,
    level: str | int = 'INFO',
) -> logging.Logger:  # 创建并配置标准 logger。
    """返回一个标准 logger，默认同时支持控制台输出和可选文件输出。

    每次调用都会清理该 logger 上已有的 handler，避免重复输出。
    控制台和文件输出使用统一的日志格式。

    Args:
        logger_name (str): logger 名称，必须是非空字符串。
        log_path (str | Path | None): 日志文件路径，可选。
            如果为 None 则只输出到控制台；
            如果是相对路径，会自动挂到项目 outputs 目录下；
            如果是绝对路径则直接使用。

            Warning:
                绝对路径不受 ``build_output_path`` 的路径穿越校验保护，
                调用方需确保路径安全。当前所有调用方均使用相对路径或
                None，绝对路径仅用于调试场景。
        level (str | int): 日志级别，可以是字符串（如 'INFO'、'DEBUG'）
            或 logging 模块的整数常量，默认为 'INFO'。

    Returns:
        logging.Logger: 配置好的 logger 实例。

    Raises:
        ValueError: 当 logger_name 为空、level 不合法、log_path 指向目录时抛出。
        TypeError: 当 level 既不是字符串也不是整数时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"logger_name": logger_name, "log_path": str(log_path) if log_path is not None else None, "level": level}, "get_logger 入口参数")
    if not is_string_like(logger_name) or not str(logger_name).strip():  # logger 名称必须是非空字符串。
        raise ValueError(f"logger_name must be a non-empty string, got {logger_name!r}")  # 名称不合法就拒绝。

    if is_string_like(level):  # 字符串形式的 level 需要先转成 logging 常量。
        normalized_level = getattr(logging, str(level).upper(), None)  # 从 logging 模块里找对应级别。
        if not isinstance(normalized_level, int):  # 找不到就说明 level 名称不合法。
            raise ValueError(f"Unsupported log level: {level!r}, expected one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")  # 直接报错。
    elif _is_integer(level):  # 也允许调用方直接传整数级别（含 numpy 2.0+ 的 np.integer）。
        if is_bool_like(level):  # bool 是 int 子类，但传 bool 作为日志级别几乎一定是误用。
            raise TypeError(f"level must be a logging level name or integer, got {type(level).__name__}")  # 拒绝 bool。
        normalized_level = int(level)  # 统一转为 Python int，兼容 numpy.integer。
    else:  # 其他类型都不支持。
        raise TypeError(f"level must be a logging level name or integer, got {type(level).__name__}")  # 类型不对就报错。

    resolved_log_path = None  # 先默认没有文件输出。
    if log_path is not None:  # 如果需要文件日志，就先检查路径。
        requested_log_path = Path(log_path)  # 统一转成 Path。
        if not requested_log_path.name:  # 没有文件名就不算有效文件路径。
            raise ValueError(f"log_path must point to a file, got directory-like path: {log_path!r}")  # 不能只给目录。
        if requested_log_path.exists() and requested_log_path.is_dir():  # 已存在且是目录时也不能当文件写。
            raise ValueError(f"log_path must point to a file, got existing directory: {requested_log_path}")  # 目录不是日志文件。

        if requested_log_path.is_absolute():  # 绝对路径直接使用。
            # 注意：绝对路径不受 build_output_path 的路径穿越校验保护，
            # 调用方需确保路径安全。当前所有调用方均使用相对路径或 None，
            # 绝对路径仅用于调试场景。
            resolved_log_path = requested_log_path  # 保留绝对地址。
        else:  # 相对路径统一挂到项目输出目录，避免日志写到意外位置。
            resolved_log_path = build_output_path(*requested_log_path.parts)  # 按项目约定构造完整输出路径。

        resolved_log_path.parent.mkdir(parents=True, exist_ok=True)  # 确保日志目录存在。
        resolved_log_path = resolved_log_path.resolve()  # 再转成绝对路径，便于稳定写入。

    logger = logging.getLogger(logger_name)  # 取或创建指定名字的 logger。
    logger.setLevel(normalized_level)  # 设置 logger 的最低日志级别。
    logger.propagate = False  # 不向根 logger 继续传播，避免重复输出。

    for handler in logger.handlers[:]:  # 先清理旧 handler，避免重复添加。
        # 重新初始化时先清理旧 handler，避免重复输出。
        logger.removeHandler(handler)  # 从 logger 上移除旧 handler。
        try:  # 关闭 handler 可能因文件已关闭等原因抛异常，需保护。
            handler.close()  # 关闭旧 handler 释放资源。
        except Exception:  # 忽略关闭时的异常，不影响后续 handler 设置。
            pass  # 静默忽略关闭异常。

    formatter = logging.Formatter(  # 统一定义日志格式。
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"  # 时间、名字、级别和消息放在一行。
    )  # formatter 创建结束。

    handler = logging.StreamHandler()  # 先创建控制台 handler。
    handler.setLevel(normalized_level)  # 控制台 handler 也使用同样级别。
    handler.setFormatter(formatter)  # 控制台 handler 使用同一个格式。
    logger.addHandler(handler)  # 把控制台 handler 挂到 logger 上。

    if resolved_log_path is not None:  # 如果要求写文件，就再加一个文件 handler。
        # 文件输出和控制台输出使用同一套格式，便于人工比对。
        # FileHandler 默认使用追加模式（mode='a'），多次运行不会覆盖旧日志。
        handler = logging.FileHandler(resolved_log_path, encoding="utf-8")  # 创建文件 handler。
        handler.setLevel(normalized_level)  # 文件 handler 级别保持一致。
        handler.setFormatter(formatter)  # 文件 handler 也用同样格式。
        logger.addHandler(handler)  # 挂上文件 handler。

    return logger  # 返回配置好的 logger。
