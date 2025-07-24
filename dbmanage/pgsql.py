import logging
import threading
import time

from concurrent_log_handler import ConcurrentRotatingFileHandler
from psycopg import OperationalError
from psycopg_pool import ConnectionPool
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from configs.config import Config


class ConnectionPoolError(Exception):
    """自定义异常，表示数据库连接池初始化或状态异常"""
    pass


# # 设置日志基本配置，级别为DEBUG或INFO
logger = logging.getLogger(__name__)
# 设置日志器级别为DEBUG
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
logger.handlers = []  # 清空默认处理器
# 使用ConcurrentRotatingFileHandler
handler = ConcurrentRotatingFileHandler(
    # 日志文件
    Config.LOG_FILE,
    # 日志文件最大允许大小为5MB，达到上限后触发轮转
    maxBytes = Config.MAX_BYTES,
    # 在轮转时，最多保留3个历史日志文件
    backupCount = Config.BACKUP_COUNT
)
# 设置处理器级别为DEBUG
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(handler)

# 数据库重试机制,最多重试3次,指数退避等待2-10秒,仅对数据库操作错误重试
@retry(stop=stop_after_attempt(3),wait=wait_exponential(multiplier=1, min=2, max=10),retry=retry_if_exception_type(OperationalError))
def test_connection(db_connection_pool: ConnectionPool) -> bool:
    with db_connection_pool.getconn() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            logger.info(f"Database connection successful: {result}")
            if result != (1, ):
                raise OperationalError("Database connection failed")
    return True


def monitor_connection_pool(db_connection_pool: ConnectionPool, interval: int = 60):
    """周期性监控连接池状态"""
    def _monitor():
        while not db_connection_pool.closed:
            try:
                stats = db_connection_pool.get_stats()
                active = stats.get("connections_in_use", 0)
                total = db_connection_pool.max_size
                logger.info(f"Connection db_connection_pool status: {active}/{total} connections in use")
                if active >= total * 0.8:
                    logger.warning(f"Connection db_connection_pool nearing capacity: {active}/{total}")
            except Exception as e:
                logger.error(f"Failed to monitor connection db_connection_pool: {e}")
            time.sleep(interval)

    monitor_thread = threading.Thread(target=_monitor, daemon=True)
    monitor_thread.start()
    return monitor_thread
