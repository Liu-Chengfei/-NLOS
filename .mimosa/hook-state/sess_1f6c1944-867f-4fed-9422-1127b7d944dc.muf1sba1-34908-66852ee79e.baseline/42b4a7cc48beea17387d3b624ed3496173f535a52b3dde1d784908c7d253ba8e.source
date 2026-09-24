from __future__ import annotations

"""日志工具（log_utils）测试模块。

文件职责：验证 get_logger 能正确创建带文件输出的日志器。

测试覆盖范围：
- 正常场景：文件日志输出
- 异常场景：非法日志级别、目录路径拒绝

被测模块：liquidloc.common.log_utils"""


import logging

import pytest

from liquidloc.common.log_utils import get_logger


def test_get_logger_writes_file_output(tmp_path):
    log_path = tmp_path / "logs" / "common.log"
    logger = get_logger("tests.common.log_utils", log_path=log_path, level="WARNING")

    try:
        logger.warning("common gate message")
        for handler in logger.handlers:
            handler.flush()

        assert logger.level == logging.WARNING
        assert len(logger.handlers) == 2
        assert log_path.exists()
        assert "common gate message" in log_path.read_text(encoding="utf-8")
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()


def test_get_logger_rejects_invalid_level():
    with pytest.raises(ValueError, match="Unsupported log level"):
        get_logger("tests.common.log_utils.invalid", level="LOUD")


def test_get_logger_rejects_directory_log_path(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    with pytest.raises(ValueError, match="log_path must point to a file"):
        get_logger("tests.common.log_utils.directory", log_path=log_dir)
