"""Top-level package for PyCMTensor."""

__author__ = """Melvin Wong"""
__version__ = "1.0.7"

from .logger import main_logger

log = main_logger.log

from .config import Config

config = Config()

import numpy as np

from .data import Data

rng = np.random.default_rng(config["seed"])
print(
    f"Python {config.info['python_version'].split(' |')[0]}",
    f"| PyCMTensor {__version__}",
)
