import logging
import os
import sys
from typing import Optional

_DEFAULT_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ZlLogger:
    """Singleton wrapper around the stdlib logging module.

    Level is read from the LOG_LEVEL env var (DEBUG/INFO/WARNING/ERROR/CRITICAL)
    at first instantiation; defaults to INFO.
    """

    _instance: Optional["ZlLogger"] = None

    def __new__(cls, name: str = "zrt") -> "ZlLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init(name)
        return cls._instance

    def _init(self, name: str) -> None:
        level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, _DATE_FORMAT))
            logger.addHandler(handler)

        self._logger = logger

    def set_level(self, level) -> None:
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        self._logger.setLevel(level)

    def debug(self, msg, *args, **kwargs) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs) -> None:
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs) -> None:
        self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs) -> None:
        self._logger.exception(msg, *args, **kwargs)


logger = ZlLogger()
