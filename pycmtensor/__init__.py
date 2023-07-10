"""Top-level package for PyCMTensor."""

__author__ = """Melvin Wong"""
__version__ = "1.3.2"

import aesara
from aesara.tensor.sharedvar import TensorSharedVariable as TensorSharedVariable
from aesara.tensor.var import TensorVariable as TensorVariable
from watermark import watermark

import pycmtensor.optimizers
import pycmtensor.scheduler
from pycmtensor.config import config

# aesara configs

aesara.config.on_unused_input = "ignore"
aesara.config.mode = "Mode"
aesara.config.allow_gc = False

# model default configs

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
    "validation_threshold",
    1.003,
    "The factor of the validation error score to meet in order to register an improvement",
)
config.add("base_learning_rate", 0.01, "The initial learning rate of the model update")
config.add(
    "max_learning_rate",
    0.1,
    "The maximum learning rate (additional option for various schedulers)",
)
config.add(
    "min_learning_rate",
    1e-5,
    "The minimum learning rate (additional option for various schedulers)",
)
config.add(
    "convergence_threshold",
    1e-4,
    "The gradient norm convergence threshold before model termination",
)
config.add(
    "optimizer",
    pycmtensor.optimizers.Adam,
    "Optimization algorithm to use for model estimation",
)

config.add(
    "lr_scheduler",
    pycmtensor.scheduler.ConstantLR,
    "Learning rate scheduler to use for model estimation",
)
config.add("lr_ExpRangeCLR_gamma", 0.5, "Gamma parameter for `ExpRangeCLR`")
config.add("lr_stepLR_factor", 0.5, "Drop step multiplier factor for `stepLR`")
config.add("lr_stepLR_drop_every", 10, "Drop learning rate every n steps for `stepLR`")
config.add("lr_CLR_cycle_steps", 16, "Steps per cycle for `CyclicLR`")
config.add("lr_PolynomialLR_power", 0.999, "Power factor for `PolynomialLR`")
config.add(
    "BFGS_warmup",
    10,
    "Discards this number of hessian matrix updates when running the `BFGS` algorithm",
)


def about():
    """Returns a `watermark.watermark` of various system information for debugging"""
    return watermark(
        python=True,
        datename=True,
        updated=True,
        packages="pycmtensor,aesara,numpy,scipy",
        machine=True,
    )
