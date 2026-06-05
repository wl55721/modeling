import functools
import logging
import sys
import time
from contextlib import contextmanager
from typing import Callable


def get_logger(name: str = "root", level: int = logging.INFO) -> logging.Logger:
    """Create or retrieve a configured logger (stdout handler)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        fmt = logging.Formatter(
            "[%(levelname)s] %(asctime)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


logger: logging.Logger = get_logger("kepler")
"""Global kepler logger — import and use directly in all modules."""


# ── automatic tracing ──────────────────────────────────────

def _fmt_val(v) -> str:
    """Summarise a value for log output — short for scalars, lengths for collections."""
    if isinstance(v, (int, float, str, bool)) or v is None:
        s = str(v)
        return s[:120] + "..." if len(s) > 120 else s
    if isinstance(v, (list, tuple, set)):
        return f"{type(v).__name__}[{len(v)}]"
    if isinstance(v, dict):
        keys = ", ".join(str(k) for k in list(v.keys())[:5])
        suffix = "..." if len(v) > 5 else ""
        return f"dict{{{keys}{suffix}}}"
    return type(v).__name__


def _fmt_args(args: tuple, kwargs: dict, skip_self: bool = True) -> str:
    """Build a compact argument description."""
    parts = []
    for a in args:
        if skip_self and len(parts) == 0:
            s = _fmt_val(a)
            if s.startswith("<") or "object" in s:
                continue  # skip 'self' reference
        parts.append(_fmt_val(a))
    for k, v in kwargs.items():
        parts.append(f"{k}={_fmt_val(v)}")
    return ", ".join(parts)


def traced(func: Callable | None = None, *, level: int = logging.DEBUG):
    """Decorator: log entry / exit + timing for the decorated function.

    Usage:
        @traced
        def my_func(x, y): ...

        @traced(level=logging.INFO)
        def my_func(x, y): ...
    """
    def _decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            arg_str = _fmt_args(args, kwargs)
            logger.log(level, "%s(%s)", fn.__qualname__, arg_str)
            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                dt = (time.perf_counter() - t0) * 1000
                logger.log(level, "%s -> %s (%.2f ms)", fn.__qualname__, _fmt_val(result), dt)
                return result
            except Exception:
                dt = (time.perf_counter() - t0) * 1000
                logger.warning("%s FAILED after %.2f ms", fn.__qualname__, dt, exc_info=True)
                raise

        return wrapper

    if func is not None:
        return _decorator(func)
    return _decorator


@contextmanager
def log_call(_logger: logging.Logger, label: str, *, level: int = logging.DEBUG):
    """Context manager for inline instrumented blocks.

    Usage:
        with log_call(log, "execute model"):
            results = execute_model(...)
        # Logs: "execute model — start" / "execute model — done (12.3 ms)"
    """
    _logger.log(level, "%s — start", label)
    t0 = time.perf_counter()
    try:
        yield
        dt = (time.perf_counter() - t0) * 1000
        _logger.log(level, "%s — done (%.2f ms)", label, dt)
    except Exception:
        dt = (time.perf_counter() - t0) * 1000
        _logger.warning("%s — FAILED after %.2f ms", label, dt, exc_info=True)
        raise
