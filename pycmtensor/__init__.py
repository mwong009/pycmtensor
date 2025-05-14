"""Top-level package for PyCMTensor.

This code snippet defines the default configuration settings for the PyCMTensor package. It sets various parameters such as batch size, seed value, maximum number of epochs, learning rate, optimizer, and learning rate scheduler.

Example Usage:
import pycmtensor.defaultconfig as defaultconfig

config = defaultconfig.config

Inputs:
No specific inputs are required for this code snippet.

Outputs:
The code snippet does not produce any outputs directly. It sets the default configuration settings for the PyCMTensor package, which can be accessed and used by other parts of the package.
"""

__author__ = """Melvin Wong"""
__version__ = "1.14.2"

import importlib

import aesara

import pycmtensor.defaultconfig as defaultconfig
import pycmtensor.optimizers as optimizers
import pycmtensor.scheduler as scheduler

# shortcuts for importing commonly used classes
from pycmtensor.dataset import Dataset
from pycmtensor.regularizers import Regularizers
from pycmtensor.run import compute, train

config = defaultconfig.config

# defaults
defaultScheduler = scheduler.ConstantLR()
defaultOptimizer = optimizers.SGD()

# aesara configs
aesara.config.on_unused_input = "ignore"
aesara.config.mode = "Mode"
aesara.config.allow_gc = False

# model default configs

USE_MIN_ERROR = 0
USE_MAX_LOG_LIKELIHOOD = 1

config.add(
    "batch_size",
    32,
    "Number of samples processed on each iteration of the model update",
)
config.add("seed", 100, "Seed value for random number generators")
config.add("max_epochs", 500, "Maximum number of model update epochs")
config.add("patience", 2000, "Minimum number of model update iterations to process")
config.add(
    "patience_increase",
    2,
    "Increase patience by this factor if model does not converge",
)
config.add(
    "likelihood_threshold",
    1.005,
    "The factor of the likelihood improvement to output a result",
)
config.add(
    "validation_threshold",
    1.2,
    "The factor of the validation error improvement",
)
config.add(
    "convergence_threshold",
    1e-4,
    "The gradient norm convergence threshold before model termination",
)
config.add("adam_weight_decay", 0.01, "Weight decay factor for AdamW optimizer")
config.add(
    "acceptance_method",
    1,
    "Best model acceptance method 1 (default): maximum loglikelihood; 0: min validation error",
)


def about():
    """Returns a `watermark.watermark` of various system information for debugging"""

    def import_wm():
        try:
            importlib.import_module("watermark")
            return True
        except ImportError:
            return False

    if import_wm():
        from watermark import watermark

        return watermark(
            python=True,
            datename=True,
            updated=True,
            packages="pycmtensor,aesara,numpy,scipy",
            machine=True,
        )
    else:
        print("{0:12s}: {1}".format("PyCMTensor", __version__))
        print("{0:12s}: {1}".format("Aesara", aesara.__version__))
