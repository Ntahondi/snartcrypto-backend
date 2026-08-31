# src/utils/production_logger.py
"""
Production logger with minimal output and structured logging
"""

import logging
import sys
from src.utils.safe_logger import SafeRotatingFileHandler
import os


def setup_production_logging():
    """
    Set up production logging with minimal console output.
    Only errors and warnings go to console. Everything else goes to file.
    """
    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Root logger - WARNING and above to console
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)

    # Console handler - only WARNING and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler - ALL logs to file
    file_handler = SafeRotatingFileHandler(
        "logs/smartcrypto.log",
        maxBytes=10_000_000,  # 10MB
        backupCount=5,
        delay=True
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Set specific modules to WARNING (reduce noise)
    logging.getLogger('src.services.model_trainer').setLevel(logging.WARNING)
    logging.getLogger('src.services.orderbook_monitor').setLevel(logging.WARNING)
    logging.getLogger('src.services.history_manager').setLevel(logging.INFO)
    logging.getLogger('src.data.collectors').setLevel(logging.WARNING)

    return root_logger