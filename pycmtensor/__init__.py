"""Top-level package for PyCMTensor."""

__author__ = """Melvin Wong"""
__version__ = "0.5.2"

import logging

default_formatter = logging.Formatter(
    "[{asctime:s}] {name:26s} {levelname:s}: {message:s}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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
    return logger


def hello():
    print(f"PyCMTensor version {__version__}")


logger = get_default_logger(__name__, level=logging.INFO)

from .configparser import PyCMTensorConfig
from .database import *
from .pycmtensor import *

config = PyCMTensorConfig()
config.generate_config_file()
config.set_num_threads()
