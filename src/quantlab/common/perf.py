"""
Performance measurement utilities
"""
import time
import tracemalloc
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Dict, Optional

from .logging import get_logger


logger = get_logger(__name__)


@contextmanager
def measure_time(name: str, log: bool = True):
    """
    Measure execution time.

    Args:
        name: Operation name
        log: Whether to log the result

    Yields:
        None
    """
    start = time.perf_counter()
    if log:
        logger.info(f"timer_start", operation=name)

    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if log:
            logger.info(f"timer_end", operation=name, elapsed_seconds=elapsed)


def timed(name: Optional[str] = None, log: bool = True):
    """
    Decorator to time function execution.

    Args:
        name: Operation name (defaults to function name)
        log: Whether to log the result
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = name or func.__name__
            with measure_time(op_name, log=log):
                return func(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def measure_memory(name: str, log: bool = True):
    """
    Measure memory usage.

    Args:
        name: Operation name
        log: Whether to log the result

    Yields:
        None
    """
    tracemalloc.start()
    if log:
        logger.info(f"memory_start", operation=name)

    try:
        yield
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if log:
            logger.info(
                f"memory_end",
                operation=name,
                current_mb=current / (1024 * 1024),
                peak_mb=peak / (1024 * 1024),
            )


class PerformanceTracker:
    """
    Track performance metrics across multiple operations.
    """

    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.memory: Dict[str, tuple] = {}

    def add_timing(self, name: str, elapsed: float) -> None:
        """Record timing."""
        if name in self.timings:
            self.timings[name] += elapsed
        else:
            self.timings[name] = elapsed

    def add_memory(self, name: str, current: float, peak: float) -> None:
        """Record memory usage."""
        self.memory[name] = (current, peak)

    def get_summary(self) -> Dict:
        """Get performance summary."""
        return {
            "timings": self.timings,
            "memory": self.memory,
            "total_time": sum(self.timings.values()),
        }

    def log_summary(self) -> None:
        """Log performance summary."""
        summary = self.get_summary()
        logger.info("perf_summary", **summary)


# Global tracker
_global_tracker: Optional[PerformanceTracker] = None


def get_tracker() -> PerformanceTracker:
    """Get global performance tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = PerformanceTracker()
    return _global_tracker


def reset_tracker() -> None:
    """Reset global performance tracker."""
    global _global_tracker
    _global_tracker = PerformanceTracker()


def track_timing(name: str):
    """
    Decorator to track function timing in global tracker.

    Args:
        name: Operation name
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                get_tracker().add_timing(name, elapsed)
        return wrapper
    return decorator
