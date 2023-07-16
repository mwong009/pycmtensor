# Installing PyCMTensor

---

To ensure complete installation including the necessary libraries, it is recommended to first install dependencies via `conda` package manager in a virtual environment, then install PyCMTensor via `pip`.

## System requirements
- Python (3.9+)
- Aesara (2.9+) - from conda-forge
- Numpy - from conda-forge
- Scipy
- Pandas

In addition, you will need:

- A C compiler compatible with your OS and Python installation. Libraries can be installed from conda-forge:
    - Linux: `gcc_linux-64` and `gxx_linux-64`
    - Windows (7 or later): `m2w64-toolchain` and `vs2019_win-64`
    - macOS (incl. M1): `Clang`
- BLAS installation
    - MKL libraries, installed through conda with `mkl-service` package
    - Openblas, default when Numpy is installed with pip, alternatively, with conda `blas` package


## Installation steps

1. [Install conda dependencies](#install-conda-dependencies)
2. [Install PyCMTensor](#install-pycmtensor-using-pip)
3. [Validate installation](#checking-your-installation)



## Install conda dependencies

Install [Miniconda](https://conda.io/miniconda.html). Select the appropriate package for your operating system.

Once you have installed conda, create a virtual environment and activate it

    :::bash
    conda create -n pycmtensor python=3.11 
    conda activate pycmtensor


Install the conda dependencies:

**Windows**
 
    :::bash
    conda install -c conda-forge mkl-service m2w64-toolchain vs2019_win-64 blas aesara -y

**macOS (incl. M1)**

    :::bash
    conda install -c conda-forge mkl-service Clang blas aesara -y

**Linux**

    :::bash
    conda install -c conda-forge mkl-service gcc_linux-64 gxx_linux-64 blas aesara -y

## Install PyCMTensor using pip

Once the conda packages have been installed, install the rest of the packages using `pip`, type:

    :::bash
    pip install pycmtensor



### Checking your installation

If PyCMTensor was installed correctly, the following should display when you run the following code in a python console:

    :::bash
    python -c "import pycmtensor; print(pycmtensor.__version__)"


Output:

    :::bash
    1.3.2

### Updating PyCMTensor

Update PyCMTensor by running the `pip install --upgrade pycmtensor` command


## Source code

Source code can be checked out from the Github repository via `git`:

    
    :::bash
    git clone git::/github.com/mwong009/pycmtensor
