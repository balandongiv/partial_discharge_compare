"""
Utility Logger Configuration for Partial Discharge Classification Pipeline

This utility module provides a streamlined logging configuration specifically designed
for utility functions and helper modules within the PD classification pipeline. It
implements a simplified logging setup that ensures consistent log formatting and
proper log level management for utility functions.

Step-by-Step Process:
1. Logger Instance Management:
   - Creates or retrieves logger instance by name
   - Prevents duplicate handler registration through handler checking
   - Ensures singleton behavior for each logger name
   - Optimizes resource usage by reusing existing loggers

2. Log Level Configuration:
   - Sets logging level to INFO for comprehensive utility tracking
   - Balances detail level between verbosity and performance
   - Captures important utility function operations and results

3. Console Handler Setup:
   - Creates StreamHandler for console output
   - Applies standardized format: "[LEVEL] logger_name: message"
   - Provides immediate feedback during utility function execution
   - Enables real-time monitoring of utility operations

4. File Handler Configuration (Optional):
   - Creates FileHandler when log_dir is specified
   - Saves logs to pipeline.log in specified directory
   - Includes timestamp in format: "YYYY-MM-DD HH:MM:SS [LEVEL] logger_name: message"
   - Enables persistent logging for utility function debugging

5. Handler Registration:
   - Adds console handler to logger instance
   - Conditionally adds file handler based on log_dir parameter
   - Prevents duplicate handlers through existence checking
   - Ensures proper cleanup and resource management

6. Format Configuration:
   - Console format: Simple, readable format for real-time monitoring
   - File format: Detailed format with timestamps for historical analysis
   - Consistent formatting across all utility modules
   - Easy parsing for log analysis tools

Usage Pattern:
- Utility functions call get_logger(__name__, log_dir=output_path / "reports")
- Logger name corresponds to utility module name for clear identification
- Log directory typically points to reports/ subdirectory
- Logs capture utility function progress, errors, and completion status

Configuration Parameters:
- name: Logger name (typically __name__ of calling utility module)
- log_dir: Optional directory for log file output
- level: Logging level (default: INFO)

Dependencies:
- logging: Python standard logging framework
- pathlib: Cross-platform path handling

Output Structure:
- Console: Real-time log messages during utility execution
- File: pipeline.log in specified directory with timestamps
- Format: Consistent formatting for easy parsing and analysis

Benefits:
- Centralized logging configuration for utility functions
- Dual output for monitoring and persistence
- Consistent formatting across all utility modules
- Easy debugging and performance analysis
- Proper resource management and cleanup
- Simplified interface for utility function logging needs
"""

from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(console)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "pipeline.log")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)
    return logger


