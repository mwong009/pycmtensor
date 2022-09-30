# Installation

## Requirements

| Module | Version    | Notes                                                                                 |
| :----- | :--------- | :------------------------------------------------------------------------------------ |
| Python | 3.9.12     | Python 3.9+ supported. Python 3.10 untested.                                          |
| Numpy  | >=1.19.0   | Older versions may be compatible                                                      |
| Aesara | >2.7.4     | Latest version of Aesara can be downloaded from https://github.com/aesara-devs/aesara |
| Scipy  | >1.7.1     |                                                                                       |
| MKL    | >=2022.0.1 | Installed through Conda                                                                            |	Installed through conda environment

## Install Conda (miniconda)

Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

Full Anaconda works fine, but Miniconda is recommmended for a minimal installation. Ensure that Conda is using at least **Python 3.9**

Next, install the required dependencies:

::::{tab-set}

:::{tab-item} Windows
```console
conda install mkl-service conda-forge::cxx-compiler conda-forge::m2w64-toolchain -y
```
:::

:::{tab-item} OSX
```console
conda install mkl-service Clang -y
```
:::

:::{tab-item} Linux/Ubuntu
```console
conda install mkl-service conda-forge::cxx-compiler -y
```
:::

::::

## Stable release installation

Once you have installed the Conda dependencies, download and install the latest 
branch of ``PyCMTensor`` from [PyPI](https://pypi.org/project/pycmtensor). 
In your conda environment, run:

```console
pip install -U pycmtensor
```

Alternatively, the latest development version is available via [Github](https://github.com/mwong009/pycmtensor). It can be installed via 

```console
pip install -U git+https://github.com/mwong009/pycmtensor.git
```

## Source code

The source code for PyCMTensor can be downloaded from the [main Github repo](https://github.com/mwong009/pycmtensor).

```console
git clone git://github.com/mwong009/pycmtensor
```