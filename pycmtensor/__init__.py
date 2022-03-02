"""Top-level package for PyCMTensor."""

__author__ = """Melvin Wong"""
__version__ = "0.5.0"

from pycmtensor.configparser import PyCMTensorConfig

config = PyCMTensorConfig()
config.generate_config_file()
config.set_num_threads()

from pycmtensor.database import *
from pycmtensor.pycmtensor import *
