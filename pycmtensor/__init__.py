"""Top-level package for PyCMTensor."""

__author__ = """Melvin Wong"""
__version__ = "1.3.2"

import aesara
from watermark import watermark

import pycmtensor.optimizers
import pycmtensor.scheduler

from .config import config
from .logger import *
from .optimizers import *
from .scheduler import *

# aesara configs

aesara.config.on_unused_input = "ignore"
aesara.config.mode = "Mode"
aesara.config.allow_gc = False

# model parameters

config.add("batch_size", 32, "Number of samples processed on model update iteration")

config.add("seed", 42069, "Seed value for random number generators")

config.add("max_steps", 1000, "Maximum number of model update steps")

config.add("patience", 200, "Minimum number of model update iterations to process")

config.add(
    "patience_increase",
    2,
    "Increase patience by this factor if model improves over the validation threshold",
)

config.add("validation_threshold", 1.003, "A factor multiplied by the validation score")

config.add("base_learning_rate", 0.01, "The initial learning rate of the model update")

config.add("max_learning_rate", 0.1, "The maximum learning rate for various schedulers")

config.add(
    "convergence_threshold",
    0.003,
    "Default convergence threshold. Teminate model estimation when average coefficient difference between steps is lower than this threshold.",
)

config.add(
    "batch_shuffle",
    False,
    "If True, shuffles the samples in each batch at each model update step",
)

config.add(
    "optimizer",
    Adam,
    f"Default gradient descent algorithm, possible options are: "
    + f", ".join(pycmtensor.optimizers.__all__),
)

config.add(
    "lr_scheduler",
    ConstantLR,
    "Default learning rate scheduler, possible options are:"
    + f", ".join(pycmtensor.scheduler.__all__),
)

config.add("lr_ExpRangeCLR_gamma", 0.5, "Default gamma value for ExpRangeCLR")

config.add("lr_stepLR_factor", 0.5, "Default step factor value for stepLR")
config.add("lr_stepLR_drop_every", 1, "Default drop every value for stepLR")

config.add("lr_CLR_cycle_steps", 16, "Default steps per cycle for CyclicLR")

config.add("lr_PolynomialLR_power", 0.999, "Default power value for PolynomialLR")


def about():
    print(
        watermark(
            python=True,
            datename=True,
            updated=True,
            packages="pycmtensor,aesara,numpy,scipy",
            machine=True,
        )
    )
