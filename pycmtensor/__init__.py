"""Top-level package for PyCMTensor."""

__author__ = """Melvin Wong"""
__version__ = "0.6.0"

from .configparser import config

config.set_num_threads()
config.generate_config_file()

from .database import *
from .pycmtensor import *
