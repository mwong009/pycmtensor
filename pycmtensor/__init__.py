"""Top-level package for PyCMTensor."""

__author__ = """Melvin Wong"""
__version__ = "1.3.2"

import numpy as np

from .config import Config, about
from .logger import *

config = Config()
rng = np.random.default_rng(config["seed"])

print(
    f"Python {config.info['python_version'].split(' |')[0]}",
    f"| PyCMTensor {__version__}",
)

from .data import *
