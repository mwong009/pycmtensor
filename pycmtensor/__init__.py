"""Top-level package for PyCMTensor."""

__author__ = """Melvin Wong"""
__version__ = "0.5.1"

from pycmtensor.configparser import PyCMTensorConfig

config = PyCMTensorConfig()
config.generate_config_file()
config.set_num_threads()

from pycmtensor.database import *
from pycmtensor.pycmtensor import *


def hello():
    print(f"PyCMTensor version {__version__}")
