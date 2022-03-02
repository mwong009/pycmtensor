# config.py

import glob
import multiprocessing
import os


def generate_config_file():
    aesararc_dir = os.path.expanduser("~")
    ld_dir = os.path.join(os.getenv("CONDA_PREFIX"), "Library", "bin")
    mkt_rt_files = glob.glob(os.path.join(ld_dir, "mkl_rt*"))
    if len(mkt_rt_files) > 0:
        mkl_rt_flag = os.path.basename(mkt_rt_files[-1])
        if mkl_rt_flag.endswith(".dll"):
            mkl_rt_flag = mkl_rt_flag[:-4]
    else:
        mkl_rt_flag = ""

    with open(os.path.join(aesararc_dir, ".aesararc"), "w") as f:
        f.write(
            "# Autogenerated by PyCMTensor\n\n"
            f"[blas]\nldflags = -L{ld_dir} -l{mkl_rt_flag}"
        )


def set_NUM_THREADS():
    num_cores = multiprocessing.cpu_count()
    os.environ["MKL_NUM_THREADS"] = str(num_cores)
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
