# Installing PyCMTensor

---

## Prerequisites

Before you begin, ensure you have met the following requirements:
- You have [miniconda](https://conda.io/miniconda.html) installed on your system

## Installation

PyCMTensor is available on Conda Forge. You can install it using the following command:

```bash
conda install -c conda-forge pycmtensor
```

For isolated development or testing, we recommend installing PyCMTensor in a virtual environment:

```bash
conda create -n pycmtensor-dev -c conda-forge pycmtensor
conda activate pycmtensor-dev
```

You can verify your installation by checking the version of PyCMTensor:

```bash
python -c "import pycmtensor as cmt; print(cmt.__version__)"
```

## Updating PyCMTensor

To update PyCMTensor to the latest version, run the following command:

```bash
conda update pycmtensor
```

## Source code

If you want to contribute to the project or prefer to build from source, you can clone the source code from the GitHub repository:

```bash
git clone https://github.com/mwong009/pycmtensor.git
```

Change your current directory to the cloned repository:

```bash
cd pycmtensor
```

Create a new Conda environment using the provided `environment.yml` file, and activate it:

```bash
conda env create -f environment.yml
conda activate pycmtensor-dev
```

This will set up a development environment with all the necessary dependencies installed. You can now start contributing to PyCMTensor or build it from source.