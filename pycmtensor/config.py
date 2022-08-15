# config.py
"""PyCMTensor config module"""
import configparser
import glob
import multiprocessing
import os
import subprocess
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
            if sys.platform != "win32":
                b = "mkl_rt"
            else:
                if b.endswith(".dll"):
                    b = b[:-4]
            blas_flags = [f"-l{os.path.basename(b)}"]
        return blas_flags
    else:
        raise NameError(
            "CONDA_PREFIX not found, please add CONDA_PREFIX to environment variables"
        )


def generate_cxx_flags_macos():
    """Finds and generates the gcc__cxxflags config option in macos

    Returns:
        list: a list of cxx flags to pass to .aesararc
    """
    sdk_path = subprocess.getoutput("xcrun --show-sdk-path")
    cxxflags = os.path.join(sdk_path, "usr", "include")
    return [f"-I{cxxflags} "]


def generate_ld_path_flags():
    """Finds and generates the blas__ldflags config option for .aesararc

    Raises:
        NameError: Raises an error if Conda is not installed.

    Returns:
        list: a list of ld_path flags prefixed with "-L"
    """
    if "CONDA_PREFIX" in os.environ:
        if sys.platform == "win32":
            ld_dir = os.path.join(os.getenv("CONDA_PREFIX"), "Library", "bin")
        else:
            ld_dir = os.path.join(os.getenv("CONDA_PREFIX"), "lib")
        ld_path_flags = [f"-L{ld_dir}"]
        return ld_path_flags
    else:
        raise NameError("Conda not installed.")


def init_aesara_rc():
    """This method gets called on initialization of the Config class.

    Generates the .aesararc config file into the user's home directory if it does not
    exist
    """
    HOMEPATH = os.path.expanduser("~")
    aesararc_config_file = os.path.join(HOMEPATH, ".aesararc")
    aesararc_config = configparser.ConfigParser()

    # global
    aesararc_config.add_section("global")
    aesararc_config["global"] = {"device": "cpu", "floatX": "float64"}
    aesararc_config["global"]["allow_gc"] = "False"  # disables garbage collector
    aesararc_config["global"]["openmp"] = "False"
    aesararc_config["global"]["optimizer"] = "fast_compile"
    aesararc_config["global"]["optimizer_including"] = "local_remove_all_assert"
    aesararc_config["global"]["optimizer_excluding"] = "inplace"

    # blas
    aesararc_config.add_section("blas")
    ldflags = "".join(f"{ld_path} " for ld_path in generate_ld_path_flags())
    ldflags += "".join(f"{blas} " for blas in generate_blas_flags())
    aesararc_config["blas"]["ldflags"] = ldflags

    # gcc
    aesararc_config.add_section("gcc")
    aesararc_config["gcc"]["cxxflags"] = "".join(
        f"-O3 -ffast-math -ftree-loop-distribution -funroll-loops -ftracer "
    )
    if sys.platform == "darwin":
        aesararc_config["gcc"]["cxxflags"] += "".join(
            f"{cxxflags} " for cxxflags in generate_cxx_flags_macos()
        )

    with open(aesararc_config_file, "w") as f:
        aesararc_config.write(f)

    return aesararc_config


def _config():
    """Defines the default model hyperparameters and config options"""
    rng = np.random.default_rng()
    config = {
        "python_version": sys.version,
        "cwd": os.getcwd(),
        "patience": 9000,
        "patience_increase": 2,
        "validation_threshold": 1.003,
        "seed": rng.integers(1, 9000),
        "base_lr": 0.0001,
        "max_lr": 0.01,
        "batch_size": 64,
        "max_epoch": 2000,
        "verbosity": "high",
        "debug": False,
        "notebook": False,
        "learning_scheduler": "ConstantLR",
        "cyclic_lr_mode": None,
        "cyclic_lr_step_size": None,
    }
    return config


def init_environment_variables():
    """Sets misc. options in the environment variables"""
    num_cores = multiprocessing.cpu_count()
    os.environ["MKL_NUM_THREADS"] = str(num_cores)
    os.environ["OMP_NUM_THREADS"] = str(num_cores)


class Config:
    """Default config class"""

    def __init__(self):
        self.config = _config()
        self.aesara_rc = init_aesara_rc()
        init_environment_variables()

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

    def set_aesararc(self, section, var, value):
        if self.aesara_rc.has_section(section):
            self.aesara_rc[section][var] = value

    def append_aesararc(self, section, var, value):
        if self.aesara_rc.has_section(section):
            self.aesara_rc[section][var] += f"{value} "
