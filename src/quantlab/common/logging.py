"""
Structured logging with run_id support
"""
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

# Global run_id
_run_id: Optional[str] = None


def set_run_id(run_id: str) -> None:
    """Set global run_id for logging."""
    global _run_id
    _run_id = run_id


def get_run_id() -> Optional[str]:
    """Get current run_id."""
    return _run_id


def setup_logging(log_dir: Optional[Path] = None, level: int = logging.INFO) -> None:
    """
    Setup structured logging.

    Args:
        log_dir: Directory to store log files (None = console only)
        level: Logging level
    """
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{get_run_id() or 'quantlab'}.log"

        handlers = [
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ]
    else:
        handlers = [logging.StreamHandler(sys.stdout)]

    processors.append(structlog.processors.JSONRenderer())

    logging.basicConfig(
        format="%(message)s",
        level=level,
        handlers=handlers,
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger."""
    return structlog.get_logger(name)


@contextmanager
def timer(name: str, logger: Optional[structlog.stdlib.BoundLogger] = None):
    """Context manager for timing operations."""
    start = time.time()
    if logger:
        logger.info(f"timer_start", operation=name)
    try:
        yield
    finally:
        elapsed = time.time() - start
        if logger:
            logger.info(f"timer_end", operation=name, elapsed_seconds=elapsed)


@contextmanager
def log_errors(logger: Optional[structlog.stdlib.BoundLogger] = None):
    """Context manager for logging exceptions."""
    try:
        yield
    except Exception as e:
        if logger:
            logger.exception("error", error=str(e))
        raise
