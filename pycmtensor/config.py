# config.py
"""PyCMTensor config module"""
import configparser
import glob
import multiprocessing
import os
import sys

import numpy as np


def generate_blas_flags():
    """Finds and generates the blas__ldflags config option for .aesararc

    Raises:
            NameError: Raises an error if Conda is not installed.

    Returns:
            list: a list of blas flags prefixed with "-l"
    """
    if "CONDA_PREFIX" in os.environ:
        if sys.platform == "win32":
            ld_dir = os.path.join(os.getenv("CONDA_PREFIX"), "Library", "bin")
        else:
            ld_dir = os.path.join(os.getenv("CONDA_PREFIX"), "lib")
        mkt_rt_bins = glob.glob(os.path.join(ld_dir, "*mkl_rt*"))
        blas_flags = []
        for b in mkt_rt_bins:
            if b.endswith(".dll"):
                b = b[:-4]
            if sys.platform != "win32":
                b = "mkl_rt"
            blas_flags = [f"-l{os.path.basename(b)}"]
        return blas_flags
    else:
        raise NameError(
            "CONDA_PREFIX not found, please add CONDA_PREFIX to environment variables"
        )


def generate_ld_path_flags():
    """Finds and generates the blas__ldflags config option for .aesararc

    Raises:
            NameError: Raises an error if Conda is not installed.

    Returns:
            list: a list of ld_path flags prefixed with "-L"
    """
    if "CONDA_PREFIX" in os.environ:
        ld_dir = os.path.join(os.getenv("CONDA_PREFIX"), "Library", "bin")
        ld_path_flags = [f"-L{ld_dir}"]
        return ld_path_flags
    else:
        raise NameError("Conda not installed.")


def init_aesararc():
    """This method gets called on initialization of the Config class.

    Generates the .aesararc config file into the user's home directory if it does not
    exist
    """
    HOMEPATH = os.path.expanduser("~")
    aesararc_config_file = os.path.join(HOMEPATH, ".aesararc")
    aesararc_config = configparser.ConfigParser()
    aesararc_config.add_section("global")
    aesararc_config["global"] = {"device": "cpu", "floatX": "float64"}
    aesararc_config.add_section("blas")
    ldflags = "".join(f"{ld_path} " for ld_path in generate_ld_path_flags())
    ldflags += "".join(f"{blas} " for blas in generate_blas_flags())
    aesararc_config["blas"]["ldflags"] = ldflags
    with open(aesararc_config_file, "w") as f:
        aesararc_config.write(f)


def _config():
    """Defines the default model hyperparameters and config options"""
    config = {
        "python_version": sys.version,
        "cwd": os.getcwd(),
        "patience": 9000,
        "patience_increase": 2,
        "validation_threshold": 1.003,
        "seed": 999,
        "base_lr": 0.0001,
        "max_lr": 0.01,
        "batch_size": 64,
        "max_epoch": 2000,
        "debug": False,
        "notebook": False,
        "learning_scheduler": "CyclicLR",
        "cyclic_lr_mode": "triangular2",
        "cyclic_lr_step_size": 8,
    }
    return config


def init_env_vars():
    """Sets misc. options in the environment variables"""
    num_cores = multiprocessing.cpu_count()
    os.environ["MKL_NUM_THREADS"] = str(num_cores)
    os.environ["OMP_NUM_THREADS"] = str(num_cores)


class Config:
    """Default config class"""

    def __init__(self):
        self.config = _config()
        init_aesararc()
        init_env_vars()

    def __repr__(self):
        rval = f"config ={{\n"
        for key, val in self.config.items():
            rval += f"    {key}: {val},\n"
        rval += f"}}"
        return rval

    def __getitem__(self, name):
        if name in self.config:
            return self.config[name]

    def __setitem__(self, name, val):
        if name in self.config:
            self.config[name] = val
        else:
            raise NameError(f"{name} not found in config file.")

    def __call__(self):
        return self.config
