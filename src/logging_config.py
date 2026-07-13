"""
Logging configuration for the handwritten menu scanner pipeline.

Sets up structured logging with console and optional file output.
Use this instead of print() statements for production-ready logging.

Usage:
    from logging_config import setup_logging, get_logger
    
    # At application startup:
    setup_logging(level="INFO", log_file="pipeline.log")
    
    # In each module:
    logger = get_logger(__name__)
    logger.info("Processing started", extra={"image_path": path})
    logger.warning("Low confidence detected", extra={"confidence": 0.3})
    logger.error("Pipeline failed", extra={"stage": "recognition"}, exc_info=True)
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Configure structured logging for the pipeline.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, only logs to console.
        format_string: Custom format string. If None, uses default format.
    """
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    formatter = logging.Formatter(format_string)
    
    # Console handler (always enabled)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.addHandler(console_handler)
    
    # Optional file handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name, typically __name__ from the calling module
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
