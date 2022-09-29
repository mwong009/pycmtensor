# Development Guide

To develop PyCMTensor development package in a local environment, e.g. to modify, add features etc., you need to set up a virtual (Conda) environment and install the project requirements. Follow the instructions to install Conda (miniconda), then start a new virtual environment with the provided ``environment_<your OS>.yml`` file.

1. Download the git project repository into a local directory
	```console
	git clone git://github.com/mwong009/pycmtensor
	cd pycmtensor
	```

## Installing the virtual environment

**Windows**

```
conda env create -f environment_windows.yml
```

**Linux**

```
conda env create -f environment_linux.yml
```

**Mac OSX**

```
conda env create -f environment_macos.yml
```

Next, activate the virtual environment and install ``poetry`` dependency manager via ``pip``

```
conda activate pycmtensor-dev
pip install poetry
```

## Install the project and development dependencies

```
poetry install -E dev
```

% Delete all dangling remote branches from local git repository.
% git fetch --prune