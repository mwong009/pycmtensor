"""Top-level package for PyCMTensor."""

__author__ = """Melvin Wong"""
__version__ = "0.6.3"

from .config import Config

config = Config()

from .database import *
from .pycmtensor import *
