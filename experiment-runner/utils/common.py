#!/usr/bin/env python3
"""
Common utility functions for experiment runner.
"""

import logging
import os
import sys
from typing import Dict, Any, Optional

def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True
) -> None:
    """
    Set up logging for the application.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Path to log file (optional)
        console: Whether to log to console (default: True)
    """
    # Create logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Create formatters
    verbose_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(verbose_formatter)
        root_logger.addHandler(file_handler)
        
    # Suppress overly verbose logs from third-party libraries
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("fabric").setLevel(logging.WARNING)
    logging.getLogger("paramiko").setLevel(logging.WARNING)
    
    logging.info("Logging initialized")

def is_valid_file(path: str) -> bool:
    """
    Check if a file exists and is readable.
    
    Args:
        path: Path to the file
        
    Returns:
        True if the file exists and is readable, False otherwise
    """
    return os.path.isfile(path) and os.access(path, os.R_OK)

def ensure_directory(path: str) -> bool:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: Path to the directory
        
    Returns:
        True if the directory exists or was created, False otherwise
    """
    if not path:
        return False
        
    if os.path.exists(path):
        return os.path.isdir(path)
        
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to create directory {path}: {e}")
        return False
