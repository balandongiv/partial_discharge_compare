"""
Logging Configuration Module for Partial Discharge Classification Pipeline

This module provides centralized logging configuration for the PD classification pipeline.
It implements a standardized logging setup that ensures consistent log formatting,
dual output (console and file), and proper log level management across all pipeline stages.

Step-by-Step Process:
1. Logger Initialization:
   - Creates or retrieves logger instance by name
   - Prevents duplicate handler registration through handler checking
   - Ensures singleton behavior for each logger name

2. Log Level Configuration:
   - Sets logging level to INFO for comprehensive pipeline tracking
   - Balances detail level between verbosity and performance
   - Captures major processing steps and important events

3. Console Handler Setup:
   - Creates StreamHandler for console output
   - Applies standardized format: "[LEVEL] logger_name: message"
   - Provides immediate feedback during pipeline execution
   - Enables real-time monitoring of processing progress

4. File Handler Configuration (Optional):
   - Creates FileHandler when log_dir is specified
   - Saves logs to pipeline.log in specified directory
   - Includes timestamp in format: "YYYY-MM-DD HH:MM:SS [LEVEL] logger_name: message"
   - Enables persistent logging for debugging and analysis

5. Handler Registration:
   - Adds console handler to logger instance
   - Conditionally adds file handler based on log_dir parameter
   - Prevents duplicate handlers through existence checking
   - Ensures proper cleanup and resource management

6. Format Configuration:
   - Console format: Simple, readable format for real-time monitoring
   - File format: Detailed format with timestamps for historical analysis
   - Consistent formatting across all pipeline modules
   - Easy parsing for log analysis tools

Usage Pattern:
- Each pipeline module calls get_logger(__name__, log_dir=output_path / "reports")
- Logger name corresponds to module name for clear identification
- Log directory typically points to reports/ subdirectory
- Logs capture processing progress, errors, and completion status

Configuration Parameters:
- name: Logger name (typically __name__ of calling module)
- log_dir: Optional directory for log file output
- level: Logging level (default: INFO)

Dependencies:
- logging: Python standard logging framework
- pathlib: Cross-platform path handling

Output Structure:
- Console: Real-time log messages during execution
- File: pipeline.log in specified directory with timestamps
- Format: Consistent formatting for easy parsing and analysis

Benefits:
- Centralized logging configuration
- Dual output for monitoring and persistence
- Consistent formatting across all modules
- Easy debugging and performance analysis
- Proper resource management and cleanup
"""

from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name:
        Logger name, usually ``__name__`` of the caller module.
    log_dir:
        Optional directory to write a log file into. If provided, a file handler
        is added in addition to the console handler.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "pipeline.log")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    return logger


