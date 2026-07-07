from __future__ import annotations

import logging

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """
    Configure application logging once at startup.

    If handlers already exist, keep the existing handler set and only adjust the
    effective log level. This avoids clobbering uvicorn or pytest capture setup.
    """

    numeric_level = getattr(logging, (level or "INFO").strip().upper(), logging.INFO)
    root_logger = logging.getLogger()

    if root_logger.handlers:
        root_logger.setLevel(numeric_level)
        return

    logging.basicConfig(level=numeric_level, format=DEFAULT_LOG_FORMAT)
