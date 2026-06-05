from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys


def _parse_log_level(raw: str) -> int:
    """Case-insensitive log level; 'warn' aliases to 'warning'."""
    canonical = raw.strip().lower()
    if canonical == "warn":
        canonical = "warning"
    valid = {"debug", "info", "warning", "error", "critical"}
    if canonical not in valid:
        raise argparse.ArgumentTypeError(
            f"invalid log level: {raw!r}  (choose from {', '.join(sorted(valid))})"
        )
    return getattr(logging, canonical.upper())


def _setup_file_logging(log_dir: str, level: int, level_name: str) -> None:
    """Attach a RotatingFileHandler to the global kepler logger and to
    the uvicorn logger family.  Also set up console handlers for
    uvicorn so the terminal stays readable."""
    from kepler.utils.log import logger

    logger.propagate = True
    logger.setLevel(level)

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "kepler.log")

    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=20 * 1024 * 1024, backupCount=10, encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(
        "[%(levelname)s] %(asctime)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # root catches kepler logger messages via propagation
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(fh)

    # ── uvicorn loggers ──
    from uvicorn.logging import DefaultFormatter, AccessFormatter

    err_handler = logging.StreamHandler(sys.stderr)
    err_handler.setFormatter(DefaultFormatter("%(levelprefix)s %(message)s"))
    err_handler.setLevel(level)

    for name in ("uvicorn.error", "uvicorn"):
        lg = logging.getLogger(name)
        lg.setLevel(level)
        lg.propagate = False
        lg.addHandler(fh)
        lg.addHandler(err_handler)

    acc_handler = logging.StreamHandler(sys.stdout)
    acc_handler.setFormatter(AccessFormatter(
        '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
    ))
    acc_handler.setLevel(level)

    access = logging.getLogger("uvicorn.access")
    access.setLevel(level)
    access.propagate = False
    access.addHandler(fh)
    access.addHandler(acc_handler)

    print(f"Logging to {log_path} (level={level_name})")


def main():
    parser = argparse.ArgumentParser(description="Kepler LLM Inference Simulator")
    parser.add_argument(
        "--log-dir", type=str, default="/tmp/kepler",
        help="Log output directory (default: /tmp/kepler, 20 MB rotation, 10 backups)",
    )
    parser.add_argument(
        "--log-level", type=_parse_log_level, default=_parse_log_level("info"),
        metavar="LEVEL",
        help="Log level: debug, info, warn(ing), error, critical (default: info)",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    level = args.log_level
    level_name = logging.getLevelName(level).lower()

    _setup_file_logging(args.log_dir, level, level_name)

    import uvicorn
    uvicorn.run(
        "kepler.web.app:app",
        host=args.host,
        port=args.port,
        log_level=level_name,
        log_config=None,  # we wired everything in _setup_file_logging
    )


if __name__ == "__main__":
    main()
