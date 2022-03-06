"""Top-level package for PyCMTensor."""

__author__ = """Melvin Wong"""
__version__ = "0.5.3"

from .configparser import config
from .database import *
from .pycmtensor import *

config.generate_config_file()
config.set_num_threads()
