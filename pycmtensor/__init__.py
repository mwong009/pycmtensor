"""Top-level package for PyCMTensor."""

__author__ = """Melvin Wong"""
__version__ = "0.7.0"

from .config import Config

config = Config()

from .database import *
from .pycmtensor import *
