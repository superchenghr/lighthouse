# utils/logger_util.py

import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
from configs.config import Config  # 假设你已将日志配置放在 Config 中


class LoggerUtil:
    @staticmethod
    def setup_logger(name: str) -> logging.Logger:
        """
        配置并返回一个带有 ConcurrentRotatingFileHandler 的 logger 实例
        :param name: logger 名称，通常使用 `__name__`
        :return: 配置好的 logger 实例
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.handlers = []  # 清空已有的处理器，防止重复添加

        # 创建 ConcurrentRotatingFileHandler，支持多进程安全写入
        handler = ConcurrentRotatingFileHandler(
            filename=Config.LOG_FILE,
            maxBytes=Config.MAX_BYTES,
            backupCount=Config.BACKUP_COUNT
        )
        handler.setLevel(logging.DEBUG)

        # 设置日志格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        # 添加处理器
        logger.addHandler(handler)

        return logger
