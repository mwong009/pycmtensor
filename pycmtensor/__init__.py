"""Top-level package for PyCMTensor."""

__author__ = """Melvin Wong"""
__version__ = "1.0.6"

from .config import Config

config = Config()

from .database import *
from .pycmtensor import *
