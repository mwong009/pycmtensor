# logger

import logging

VERBOSITY_DEFAULT = logging.WARNING

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


class PyCMTensorError(Exception):
    """Default exception handler"""

    pass


def set_level(level):
    main_logger.setLevel(level)


def get_effective_level():
    return main_logger.getEffectiveLevel()


def get_console_handler():
    handler = logging.StreamHandler()
    handler.setFormatter(default_formatter)
    return handler


def get_file_handler():
    handler = logging.StreamHandler()
    handler.setFormatter(default_formatter)
    return handler


def get_default_logger(name, level):
    logger = logging.getLogger(name)
    logger.addHandler(get_console_handler())
    logger.setLevel(level)
    logger.propagate = False
    return logger


default_formatter = logging.Formatter(
    "[{asctime:s}] {levelname:s}: {message:s}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
)

main_logger = get_default_logger("pycmtensor", VERBOSITY_DEFAULT)

debug = main_logger.debug
info = main_logger.info
warning = main_logger.warning
error = main_logger.error
critical = main_logger.critical
exception = main_logger.exception
log = main_logger.log
