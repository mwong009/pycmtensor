# logger.py
"""PyCMTensor logger module"""
import logging

__all__ = [
    "set_level",
    "get_effective_level",
    "log",
    "debug",
    "info",
    "warning",
    "error",
    "critical",
]

VERBOSITY_DEFAULT = logging.INFO

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


def set_level(level):
    """Set the level of the logger"""
    main_logger.setLevel(level)


def get_effective_level():
    """Gets the level of the logger"""
    return main_logger.getEffectiveLevel()


def _get_console_handler():
    handler = logging.StreamHandler()
    handler.setFormatter(default_formatter)
    return handler


def _get_file_handler():
    handler = logging.StreamHandler()
    handler.setFormatter(default_formatter)
    return handler


def get_default_logger(name, level):
    logger = logging.getLogger(name)
    logger.addHandler(_get_console_handler())
    logger.setLevel(level)
    logger.propagate = False
    return logger


default_formatter = logging.Formatter(
    "[{asctime:s}] {levelname:s}: {message:s}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
)

main_logger = get_default_logger("pycmtensor", VERBOSITY_DEFAULT)
log = main_logger.log
debug = main_logger.debug
info = main_logger.info
warning = main_logger.warning
error = main_logger.error
critical = main_logger.critical
