# Installation

## Requirements

| Module | Version    | Notes                                                                                 |
| :----- | :--------- | :------------------------------------------------------------------------------------ |
| Python | 3.9.12     | Python 3.9+ supported. Python 3.10 untested.                                          |
| Numpy  | >=1.19.0   | Older versions may be compatible                                                      |
| Aesara | >2.7.4     | Latest version of Aesara can be downloaded from https://github.com/aesara-devs/aesara |
| Scipy  | >1.7.1     |                                                                                       |
| MKL    | >=2022.0.1 | Install through Conda                                                                            |	Installed through conda environment

## Installing through Conda (miniconda)

Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

Full Anaconda works fine, but Miniconda is recommmended for a minimal installation. 
Ensure that Conda is using at least **Python 3.9**

Once Conda is installed, install the required dependencies:

::::{tab-set}

:::{tab-item} Windows
```console
conda install mkl-service conda-forge::cxx-compiler conda-forge::m2w64-toolchain
```
:::

:::{tab-item} OSX
```console
conda install mkl-service Clang
```
:::

:::{tab-item} Linux/Ubuntu
```console
conda install mkl-service conda-forge::cxx-compiler
```
:::

::::

### Stable release installation

Once you have installed the Conda dependencies, download and install the latest 
branch of ``PyCMTensor`` from [PyPI](https://pypi.org/project/pycmtensor). 
In your conda environment, run:

```console
pip install -U pycmtensor 
```

The above is the preferred method to install ``PyCMTensor``, as it will always 
install the most recent stable release.
Alternatively, if you want the development version, pip install from the Github 
repository:

```console
pip install git+https://github.com/mwong009/pycmtensor.git@develop -U
```

## Source code

The source code for PyCMTensor can be downloaded from the [main Github repo](https://github.com/mwong009/pycmtensor).

```console
git clone git://github.com/mwong009/pycmtensor
```

### Conda development environment

To develop ``PyCMTensor`` in a local environment, you need to set up a virtual (Conda)
environment and install the project requirements. Follow the above instructions to 
install Conda (miniconda), then start a new virtual environment with the 
provided ``environment_<your OS>.yml`` file.

::::{tab-set}

:::{tab-item} Windows
```console
conda env create -f environment_windows.yml
```
:::

:::{tab-item} OSX
```console
conda env create -f environment_macos.yml
```
:::

:::{tab-item} Linux/Ubuntu
```console
conda env create -f environment_linux.yml
```
:::

::::

Next, activate the virtual environment and install ``poetry`` dependency manager via 
``pip``.

```console
conda activate pycmtensor-dev
(pycmtensor-dev) pip install poetry
```

### Install 
Install the project and development dependencies

```console
(pycmtensor-dev) poetry install -E dev
```