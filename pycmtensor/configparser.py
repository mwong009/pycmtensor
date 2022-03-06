# config.py

import glob
import multiprocessing
import os
import sys

from pycmtensor import logger as log


class PyCMTensorConfig:
    """Object that holds the configuration settings."""

    def __init__(self):
        self._config = {
            "patience": 9000,
            "patience_increase": 2,
            "validation_threshold": 1.003,
            "cwd": os.getcwd(),
            "python_version": sys.version,
            "seed": 999,
            "cyclic_lr_step_size": 8,
            "base_lr": 0.0001,
            "max_lr": 0.01,
            "max_epoch": 2000,
            "batch_size": 64,
            "cyclic_lr_mode": "triangular2",
        }

    def __getitem__(self, item):
        return self._config[item]

    def __setitem__(self, key, val):
        self._config[key] = val

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        return "".join(f"{key}: {val}\n" for key, val in self._config.items())

    def generate_config_file(self):
        ld_dir = os.path.join(os.getenv("CONDA_PREFIX"), "Library", "bin")
        mkt_rt_files = glob.glob(os.path.join(ld_dir, "mkl_rt*"))
        if len(mkt_rt_files) > 0:
            mkl_rt_flag = os.path.basename(mkt_rt_files[-1])
            if mkl_rt_flag.endswith(".dll"):
                mkl_rt_flag = mkl_rt_flag[:-4]
        else:
            mkl_rt_flag = ""

        aesararc_dir = os.path.expanduser("~")
        aesara_config = os.path.join(aesararc_dir, ".aesararc")
        if os.path.isfile(aesara_config):
            log.debug(".aesararc file found in user dir")
        else:

            with open(os.path.join(aesararc_dir, ".aesararc"), "w") as f:
                f.write(
                    "# Autogenerated by PyCMTensor\n\n"
                    f"[blas]\nldflags = -L{ld_dir} -l{mkl_rt_flag}"
                )

        self["LD_DIR"] = ld_dir
        self["MKL_RT_FLAG"] = mkl_rt_flag

    def set_num_threads(self):
        num_cores = multiprocessing.cpu_count()
        os.environ["MKL_NUM_THREADS"] = str(num_cores)
        os.environ["OMP_NUM_THREADS"] = str(num_cores)

        self["MKL_NUM_THREADS"] = num_cores
        self["OMP_NUM_THREADS"] = num_cores


config = PyCMTensorConfig()
