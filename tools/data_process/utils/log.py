import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

"""
main.py 初始化日志
import logging
from utils.log import setup_logger
current_time = datetime.datetime.now().strftime("%m_%d_%H_%M")
setup_logger(log_name=f"{args.dataset_name}_{current_time}", rank=rank)
logging.info("主程序开始执行...")

other.py 使用日志
import logging
logger = logging.getLogger(__name__)
logger.info("hh")
"""

def setup_logger(
    log_name: str = "my_app",
    log_dir: str = "logs",
    level: int = logging.INFO,
    rank: int = 0
) -> None:
    """
    - 所有 Rank: 都会将 WARNING 及以上级别的日志输出到 控制台。
    - 所有 Rank: 都会创建自己独立的日志文件 (log_name_rank_R.log)。
    - Rank 0 的文件和控制台会记录 INFO 及以上级别的所有日志。
    - 其他 Rank 的文件和控制台只记录 WARNING 及以上级别的日志。
    """
    logger = logging.getLogger()

    # 清除已有处理器
    if logger.hasHandlers():
        logger.handlers.clear()

    # 根据 rank 设置不同的日志级别
    log_level = level if rank == 0 else logging.WARNING
    logger.setLevel(log_level)

    log_format = logging.Formatter(f'%(asctime)s - RANK {rank} - %(name)s - %(levelname)s - %(message)s')

    # 为所有 rank 创建文件处理器、
    log_path = Path(log_dir) / log_name
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 在日志文件名中加入 rank ID
    log_file = log_path / f"rank{rank}.log"
    
    file_handler = RotatingFileHandler(
        log_file, maxBytes=50*1024*1024, backupCount=50, encoding='utf-8'
    )
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)

    if rank == 0:
        logging.info(f"Logger initialized for all ranks. Log file pattern: {log_path}/{log_name}/rank*.log")
