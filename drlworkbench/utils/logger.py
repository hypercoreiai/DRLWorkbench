"""Centralized logging configuration for DRLWorkbench."""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler


def setup_logger(
    name: str = "drlworkbench",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    log_to_console: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Set up a logger with console and/or file handlers.
    
    Parameters
    ----------
    name : str, default "drlworkbench"
        Name of the logger.
    level : int, default logging.INFO
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file : Path, optional
        Path to log file. If None, file logging is disabled.
    log_to_console : bool, default True
        Whether to log to console.
    max_bytes : int, default 10MB
        Maximum size of log file before rotation.
    backup_count : int, default 5
        Number of backup log files to keep.
        
    Returns
    -------
    logging.Logger
        Configured logger instance.
        
    Examples
    --------
    >>> logger = setup_logger("my_module", level=logging.DEBUG)
    >>> logger.info("This is an info message")
    >>> logger.error("This is an error message")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.
    
    Parameters
    ----------
    name : str
        Name of the logger.
        
    Returns
    -------
    logging.Logger
        Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Set up with defaults if not already configured
        return setup_logger(name)
    return logger
