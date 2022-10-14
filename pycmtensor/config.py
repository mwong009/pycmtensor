# config.py
"""PyCMTensor config module"""
import configparser
import glob
import multiprocessing
import os
import subprocess
import sys

import numpy as np

from .logger import log
from .scheduler import *

__all__ = ["Config"]


def generate_blas_flags():
    """Finds and generates the blas_flags config option for .aesararc"""
    if "CONDA_PREFIX" not in os.environ:
        raise NameError("CONDA_PREFIX not found in environment variables")

    if sys.platform == "win32":
        ld_dir = os.path.join(os.getenv("CONDA_PREFIX"), "Library", "bin")
    else:
        ld_dir = os.path.join(os.getenv("CONDA_PREFIX"), "lib")

    flags = [f"-L{ld_dir}"]
    mkt_rt_bins = glob.glob(os.path.join(ld_dir, "*mkl_rt*"))
    for b in mkt_rt_bins:
        if sys.platform != "win32":
            b = "mkl_rt"
        if b.endswith(".dll"):
            b = b[:-4]
        mkl_flags = [f"-l{os.path.basename(b)}"]
    flags.extend(mkl_flags)
    return flags


def generate_cxx_flags_macos():
    """Finds and generates the gcc__cxxflags config option in macos"""
    sdk_path = subprocess.getoutput("xcrun --show-sdk-path")
    cxxflags = os.path.join(sdk_path, "usr", "include")
    return [f"-I{cxxflags}"]


def init_aesara_rc():
    """Generates the .aesararc config file in the user's home directory"""
    HOMEPATH = os.path.expanduser("~")
    conf_file = os.path.join(HOMEPATH, ".aesararc")
    aesara_rc = configparser.ConfigParser()

    # section global
    aesara_rc.add_section("global")
    aesara_rc["global"] = {"device": "cpu", "floatX": "float64"}
    aesara_rc["global"]["allow_gc"] = "False"
    aesara_rc["global"]["on_unused_input"] = "ignore"
    aesara_rc["global"]["openmp"] = "True"
    aesara_rc["global"]["optimizer"] = "fast_compile"
    aesara_rc["global"]["cycle_detection"] = "fast"
    aesara_rc["global"]["optimizer_including"] = "local_remove_all_assert"

    # section blas
    aesara_rc.add_section("blas")
    ldflags = " ".join(f"{flag}" for flag in generate_blas_flags())
    aesara_rc["blas"]["ldflags"] = ldflags

    # section gcc
    aesara_rc.add_section("gcc")

    if sys.platform == "darwin":
        flags = " ".join(f"{flag}" for flag in generate_cxx_flags_macos())
        aesara_rc["gcc"]["cxxflags"] = flags
    else:
        aesara_rc["gcc"]["cxxflags"] = " ".join([""])
        # ["-03", "-ffast-math", "-funroll-loops", "-ftracer"]

    with open(conf_file, "w") as f:
        aesara_rc.write(f)

    return aesara_rc


def init_environment_variables():
    """Sets misc. options in the environment variables"""
    num_cores = multiprocessing.cpu_count()
    os.environ["MKL_NUM_THREADS"] = str(num_cores)
    os.environ["OMP_NUM_THREADS"] = str(num_cores)


class Config:
    def __init__(self):
        """Class object to store config and hyperparameters"""
        self.rng = np.random.default_rng()
        self.info = {
            "python_version": sys.version,
            "directory": os.getcwd(),
        }
        self.hyperparameters = {
            "seed": int(self.rng.integers(1, 9000)),
            "patience": 4000,
            "patience_increase": 2,
            "validation_threshold": 1.005,
            "base_learning_rate": 0.01,
            "max_learning_rate": None,
            "batch_size": 250,
            "max_steps": 1000,
            "clr_cycle_steps": 16,
            "clr_gamma": None,
            "batch_shuffle": False,
        }
        self.hyperparameters["lr_scheduler"] = ConstantLR(self["base_learning_rate"])
        self.aesara_rc = init_aesara_rc()
        init_environment_variables()

    def __print__(self):
        rval = f"\nhyperparameters\n---------------\n"
        for key, val in self.hyperparameters.items():
            rval += f"{key:<24}: {val}\n"
        return rval

    def __getitem__(self, name: str):
        if name in self.hyperparameters:
            return self.hyperparameters[name]

    def __setitem__(self, name: str, val):
        if name in self.hyperparameters:
            self.hyperparameters[name] = val
        else:
            raise NameError(f"hyperparameter {name} is not a valid option")

    def __call__(self):
        return self.hyperparameters

    def set_hyperparameter(self, key: str, value):
        """Helper command to set hyperparameters to ``key: value``"""
        self.hyperparameters[key] = value
        log(10, f"set {key}={value}")

    def set_lr_scheduler(self, scheduler):
        """Sets the config option ``lr_scheduler`` and ``base_learning rate``"""
        if not isinstance(scheduler, Scheduler):
            raise TypeError(
                f"{type(scheduler)} is not a {Scheduler} instance, perhaps missing arguments?"
            )
        if not hasattr(scheduler, "lr"):
            raise TypeError(f"{scheduler} is not a valid learning rate scheduler")
        self["lr_scheduler"] = scheduler
        self["base_learning_rate"] = scheduler.lr

        if hasattr(scheduler, "max_lr"):
            self["max_learning_rate"] = scheduler.max_lr
        else:
            self["max_learning_rate"] = None
        if hasattr(scheduler, "cycle_steps"):
            self["clr_cycle_steps"] = scheduler.cycle_steps
        else:
            self["clr_cycle_steps"] = None
        if hasattr(scheduler, "gamma"):
            self["clr_gamma"] = scheduler.gamma
        else:
            self["clr_gamma"] = None

    def check_values(self):
        """Checks validity of hyperparameter values"""
        assert isinstance(self["seed"], (int, np.int64))
        assert isinstance(self["batch_shuffle"], bool)
        assert (self["clr_gamma"] is None) or (isinstance(self["clr_gamma"], float))
