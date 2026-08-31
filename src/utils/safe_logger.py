"""
Safe Logger - Production-Grade Logging for SmartCrypto AI v3.0.0
Filters repetitive background noise in production while preserving 
trading signals, position executions, risk alerts, and errors.
"""

import logging
import sys
import re
import platform
from logging.handlers import RotatingFileHandler
from pathlib import Path


class ProductionLogFilter(logging.Filter):
    """Filters out routine background noise in production mode"""
    
    # Substrings to suppress in production logs
    NOISE_PATTERNS = [
        "Complete data fetched for",
        "Data fetched from binance",
        "ATR14 available: mean=",
        "Connecting WebSocket for",
        "Connecting OrderBook WebSocket for",
        "OrderBook WebSocket connected for",
        "WebSocket connected for",
        "Loaded 1000 records for",
        "Next hourly candle evaluation scheduled in",
        "Cached 0 signals",
        "Cached 44 signals",
        "Cached 5 signals",
        "DataStorage initialized",
        "Database initialized at",
        "HistoryManager initialized",
        "Order Book Monitor initialized",
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        # Always pass WARNING, ERROR, and CRITICAL messages
        if record.levelno >= logging.WARNING:
            return True
            
        msg = record.getMessage()
        
        # Suppress routine background patterns
        for noise in self.NOISE_PATTERNS:
            if noise in msg:
                return False
                
        return True


class SafeFormatter(logging.Formatter):
    """Formatter that handles emojis safely on Windows terminals"""
    
    EMOJI_PATTERN = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U00002500-\U00002BFF"
        u"\U00002100-\U00002149"
        "]+",
        flags=re.UNICODE
    )
    
    EMOJI_MAP = {
        '✅': '[OK]',
        '❌': '[ERROR]',
        '⚠️': '[WARN]',
        '🚀': '[START]',
        '📊': '[DATA]',
        '💰': '[FINANCE]',
        '📈': '[UP]',
        '📉': '[DOWN]',
        '🎯': '[TARGET]',
        '🔄': '[REFRESH]',
        '💾': '[SAVE]',
        '📁': '[FOLDER]',
        '🔍': '[SEARCH]',
        '🔧': '[TOOL]',
        '🧹': '[CLEAN]',
        '🎨': '[UI]',
        '🕒': '[TIME]',
        '⏭️': '[SKIP]',
        '🏋️': '[TRAIN]',
        '🔒': '[LOCK]',
        '📥': '[DOWNLOAD]',
        '📡': '[SIGNAL]',
        '📚': '[HISTORY]',
    }
    
    def __init__(self, fmt=None, datefmt=None, style='%', validate=True):
        super().__init__(fmt, datefmt, style, validate)
        self.is_windows = platform.system() == 'Windows'
        
    def format(self, record):
        msg = super().format(record)
        
        if self.is_windows and getattr(sys.stdout, 'encoding', '').lower() not in ['utf-8', 'utf8']:
            for emoji, replacement in self.EMOJI_MAP.items():
                if emoji in msg:
                    msg = msg.replace(emoji, replacement)
            
            msg = self.EMOJI_PATTERN.sub('', msg)
            msg = re.sub(r'\s+', ' ', msg)
            
        return msg


class SafeRotatingFileHandler(RotatingFileHandler):
    """
    Windows-safe RotatingFileHandler that gracefully handles PermissionError 
    during file rotation without raising errors or polluting stderr.
    """
    def shouldRollover(self, record):
        try:
            return super().shouldRollover(record)
        except (PermissionError, OSError):
            return False

    def doRollover(self):
        try:
            super().doRollover()
        except (PermissionError, OSError):
            pass


class SafeLogger:
    """Safe logger factory with noise filtering and Windows-compatible emoji formatting"""
    
    @staticmethod
    def setup_logging(
        name: str = "smartcrypto",
        log_file: str = "logs/smartcrypto.log",
        console_level: str = "INFO",
        file_level: str = "INFO",
        max_bytes: int = 10_000_000,
        backup_count: int = 5,
        enable_noise_filter: bool = True
    ):
        Path("logs").mkdir(exist_ok=True)
        
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.handlers.clear()
        
        # Suppress third-party library noise
        for third_party in ["urllib3", "httpx", "ccxt", "websockets", "asyncio", "tensorflow"]:
            logging.getLogger(third_party).setLevel(logging.WARNING)

        prod_filter = ProductionLogFilter()

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, console_level.upper(), logging.INFO))
        console_formatter = SafeFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        if enable_noise_filter:
            console_handler.addFilter(prod_filter)
        root_logger.addHandler(console_handler)
        
        # File Handler (UTF-8, max 10MB x 5 rotation files, safe on Windows)
        file_handler = SafeRotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8',
            delay=True
        )
        file_handler.setLevel(getattr(logging, file_level.upper(), logging.INFO))
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        if enable_noise_filter:
            file_handler.addFilter(prod_filter)
        root_logger.addHandler(file_handler)
        
        return logging.getLogger(name)

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


def setup_logging(*args, **kwargs):
    return SafeLogger.setup_logging(*args, **kwargs)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# Auto-configure root logger on import
setup_logging()