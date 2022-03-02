"""Top-level package for PyCMTensor."""

__author__ = """Melvin Wong"""
__version__ = "0.5.0b0"

from pycmtensor import config

config.generate_config_file()
config.set_NUM_THREADS()

from pycmtensor.database import *
from pycmtensor.pycmtensor import *
