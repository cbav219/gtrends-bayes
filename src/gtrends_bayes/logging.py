"""Structured logger setup. Library code should use ``get_logger(__name__)``.

Notebooks may print freely; library modules must not.
"""

from __future__ import annotations

import logging
import os

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_LEVEL = os.environ.get("GTRENDS_BAYES_LOG_LEVEL", "INFO").upper()

_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    root = logging.getLogger("gtrends_bayes")
    root.setLevel(_DEFAULT_LEVEL)
    if not root.handlers:
        root.addHandler(handler)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the ``gtrends_bayes`` root.

    Parameters
    ----------
    name : str
        Usually ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        A logger inheriting handlers and level from the package root.
    """
    _configure_root()
    if not name.startswith("gtrends_bayes"):
        name = f"gtrends_bayes.{name}"
    return logging.getLogger(name)
