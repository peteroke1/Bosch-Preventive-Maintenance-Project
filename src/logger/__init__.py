import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path

# Constants for log configuration
LOG_DIR = "logs"
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3 # nUMBER OF BACKUP LOG TO KEEP

#Construct project root path safely
BASE_DIR = Path(__file__).resolve().parent.parent.parent
log_dir_path = BASE_DIR / LOG_DIR
os.makedirs(log_dir_path, exist_ok=True)
log_file_path = log_dir_path / LOG_FILE


def configure_logger():
    """
    Configures the logger with a rotating file handler.
    """
    #Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels of logs

    # Prevent duplicate handlers if this function is called multiple times
    if logger.handlers:
        return logger
    
    # Define Formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
    )

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file_path, 
        maxBytes=MAX_LOG_SIZE, 
        backupCount=BACKUP_COUNT
    )

    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # Log all levels to file

    # Console handler for INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)  # Log INFO and above to console

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger