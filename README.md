# PyCMTensor

![Licence](https://img.shields.io/badge/Licence-MIT-blue)
![](https://img.shields.io/pypi/pyversions/pycmtensor)
[![PyPI version](https://badge.fury.io/py/pycmtensor.svg)](https://badge.fury.io/py/pycmtensor)
[![Documentation Status](https://readthedocs.org/projects/pycmtensor/badge/?version=latest)](https://pycmtensor.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/mwong009/pycmtensor/branch/master/graph/badge.svg?token=LFwgggDyjS)](https://codecov.io/gh/mwong009/pycmtensor)

[![Tests](https://github.com/mwong009/pycmtensor/actions/workflows/tests.yml/badge.svg)](https://github.com/mwong009/pycmtensor/actions/workflows/tests.yml)
[![CodeQL](https://github.com/mwong009/pycmtensor/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/mwong009/pycmtensor/actions/workflows/codeql-analysis.yml)
[![Publish](https://github.com/mwong009/pycmtensor/actions/workflows/publish.yml/badge.svg)](https://github.com/mwong009/pycmtensor/actions/workflows/publish.yml)
[![DOI](https://zenodo.org/badge/460802394.svg)](https://zenodo.org/badge/latestdoi/460802394)

A tensor-based discrete choice modelling Python package. 

## Citation

Cite as:

    @software{melvin_wong_2022_7249280,
      author       = {Melvin Wong},
      title        = {mwong009/pycmtensor: v1.3.1},
      year         = 2022,
      version      = {v1.3.1},
      doi          = {10.5281/zenodo.7249280},
      url          = {https://doi.org/10.5281/zenodo.7249280}
    }

## Table of contents


- [PyCMTensor](#pycmtensor)
	- [Citation](#citation)
	- [Table of contents](#table-of-contents)
	- [About PyCMTensor](#about-pycmtensor)
	- [Features](#features)
- [Quick start](#quick-start)
	- [Installation](#installation)
- [Usage](#usage)
	- [Simple example: Swissmetro dataset](#simple-example-swissmetro-dataset)
	- [Results](#results)
- [Development](#development)
	- [Installing the virtual environment](#installing-the-virtual-environment)
	- [Install the project and development dependencies](#install-the-project-and-development-dependencies)

## About PyCMTensor

PyCMTensor is a discrete choice modelling development tool on deep learning libraries, enabling development of complex models using deep neural networks.
PyCMTensor is build on [Aesara](https://github.com/aesara-devs/aesara), a tensor library which uses features commonly found in deep learning packages such as ``Tensorflow`` and ``Keras``.
``Aesara`` was chosen as the back end mathematical library because of its hackable, open-source nature.
Users of [Biogeme](https://biogeme.epfl.ch) would be familiar with the syntax of PyCMTensor.

PyCMTensor improves on [Biogeme](https://biogeme.epfl.ch) in situations where much more complex models are necessary, for example, integrating neural networks into discrete choice models.
PyCMTensor also include the ability to estimate models using 1st order stochastic gradient descent methods by default, such as Nesterov Accelerated Gradient (NAG), Adaptive momentum (ADAM), or RMSProp.

## Features

* Estimate complex choice models with neural networks using deep learning algorithms
* Combines traditional econometric models (e.g. Multinomial Logit) with deep learning models (e.g. ResNets)
* Shares similar programming syntax with ``Biogeme``, allowing easy transition between models
* Uses tensor features found in the ``Aesara`` library

---

# Quick start

## Installation

1. Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

	Full Anaconda works fine, but Miniconda is recommmended for a minimal installation. Ensure that Conda is using at least **Python 3.9**

	Next, install the required dependencies:

	**Windows**

	```
	conda install mkl-service conda-forge::cxx-compiler conda-forge::m2w64-toolchain
	```
	**Linux**

	```
	conda install mkl-service conda-forge::cxx-compiler
	```

	**Mac OSX**

	```
	conda install mkl-service Clang
	```

2. Install the ``PyCMTensor`` package

	PyCMTensor is available on PyPi https://pypi.org/project/pycmtensor/. It can be installed with ``pip``

	```
	pip install -U pycmtensor==1.3.1
	```

	Alternatively, the latest development version is available via [Github](https://github.com/mwong009/pycmtensor). It can be installed via 

	```
	pip install -U git+https://github.com/mwong009/pycmtensor.git
	```

For more information about installing, see [Installation](https://pycmtensor.readthedocs.io/en/latest/installation.html).

# Usage

PyCMTensor uses syntax very similar to ``Biogeme``. Users of ``Biogeme`` should be familiar with the syntax.
Make sure you are using the correct Conda environment and/or the required packages are installed.

## Simple example: Swissmetro dataset

1. Start an interactive session (e.g. ``IPython`` or Jupyter Notebook) and import the ``PyCMTensor`` package:
	```python
	import pycmtensor as cmt
	import pandas as pd
	```

	Several submodules to include:
	```python
	from pycmtensor.expressions import Beta # Beta class for model parameters
	from pycmtensor.models import MNL  # MNL model
	from pycmtensor.statistics import elasticities  # For calculating elasticities
	```

	For a full list of submodules and description, refer to [API Reference](https://pycmtensor.readthedocs.io/en/latest/autoapi/index.html).
	Using the [swissmetro dataset](https://biogeme.epfl.ch/data.html), we define a simple MNL model. 


> :warning: Note: The following is a replication of the results from Biogeme using the ``Adam`` optimization method with constant learning rate.


1. Import the dataset and perform some data cleaning
	```python
	swissmetro = pd.read_csv("swissmetro.dat", sep="\t")
	swissmetro.drop(swissmetro[swissmetro["CHOICE"] == 0].index, inplace=True)
	swissmetro["CHOICE"] -= 1  # set the first choice index to 0
	db = cmt.Data(df=swissmetro, choice="CHOICE")
	db.autoscale_data(except_for=["ID", "ORIGIN", "DEST"])  # scales dataset
	db.split_db(split_frac=0.8)  # split dataset into train/valid sets
	```

2. Initialize the model parameters and specify the utility functions and availability conditions
	```python
	b_cost = Beta("b_cost", 0.0, None, None, 0)
	b_time = Beta("b_time", 0.0, None, None, 0)
	asc_train = Beta("asc_train", 0.0, None, None, 0)
	asc_car = Beta("asc_car", 0.0, None, None, 0)
	asc_sm = Beta("asc_sm", 0.0, None, None, 1)

	U_1 = b_cost * db["TRAIN_CO"] + b_time * db["TRAIN_TT"] + asc_train
	U_2 = b_cost * db["SM_CO"] + b_time * db["SM_TT"] + asc_sm
	U_3 = b_cost * db["CAR_CO"] + b_time * db["CAR_TT"] + asc_car

	# specify the utility function and the availability conditions
	U = [U_1, U_2, U_3]  # utility
	AV = [db["TRAIN_AV"], db["SM_AV"], db["CAR_AV"]]  # availability
	``` 

3. Define the Multinomial Logit model
	```python
	mymodel = MNL(db, locals(), U, AV, name="MNL")
	```

4. Train the model and generate model statistics (Optionally, you can also set the training hyperparameters)
	```python
	mymodel.config.set_hyperparameter("max_steps", 200)  # set the max number of train steps
	mymodel.config.set_hyperparameter("batch_size", 128)  # set the training batch size
	mymodel.train(db)  # run the model training on the dataset `db`
	```

## Results
The following model functions outputs the statistics, results of the model, and model training

1. **Model estimates**
	```Python
	print(mymodel.results.beta_statistics())
	```

	Output:
	```
	              value   std err     t-test   p-value rob. std err rob. t-test rob. p-value
	asc_car   -0.665638  0.044783 -14.863615       0.0     0.176178    -3.77821     0.000158
	asc_sm          0.0         -          -         -            -           -            -
	asc_train -1.646826  0.048099 -34.238218       0.0     0.198978   -8.276443          0.0
	b_cost     0.024912   0.01943   1.282135  0.199795     0.016413    1.517851     0.129052
	b_time    -0.313186  0.049708  -6.300485       0.0     0.208239   -1.503979     0.132587
	```

2. **Training results**
	```Python
	print(mymodel.results.model_statistics())
	```

	Output:
	```
	                                          value
	Number of training samples used          8575.0
	Number of validation samples used        2143.0
	Init. log likelihood               -8874.438875
	Final log likelihood                -7513.22967
	Accuracy                                 59.26%
	Likelihood ratio test                2722.41841
	Rho square                             0.153385
	Rho square bar                         0.152822
	Akaike Information Criterion       15036.459339
	Bayesian Information Criterion      15071.74237
	Final gradient norm                    0.007164
	```

3. **Correlation matrix**
	```Python
	print(mymodel.results.model_correlation_matrix())
	```

	Output:
	```
	             b_cost    b_time  asc_train   asc_car
	b_cost     1.000000  0.209979   0.226737 -0.028335
	b_time     0.209979  1.000000   0.731378  0.796144
	asc_train  0.226737  0.731378   1.000000  0.664478
	asc_car   -0.028335  0.796144   0.664478  1.000000
	```

4. **Elasticities**
	```Python
	print(elasticities(mymodel, db, 0, "TRAIN_TT"))  # CHOICE:TRAIN (0) wrt TRAIN_TT
	```

	Output:
	```
	[-0.06813523 -0.01457346 -0.0555597  ... -0.03453162 -0.02809382 -0.02343637]
	```

5. **Choice probability predictions**
	```Python
	print(mymodel.predict(db, return_choices=False))
	```

	Output:
	```
	[[0.12319342 0.54372904 0.33307754]
	[0.12267997 0.54499504 0.33232499]
	[0.12354587 0.54162143 0.3348327 ]
	...
	[0.12801816 0.5201341  0.35184774]
	[0.1271984  0.51681635 0.35598525]
	[0.12881032 0.51856181 0.35262787]]
	```

---

# Development

(Optional) To develop PyCMTensor development package in a local environment, e.g. to modify, add features etc., you need to set up a virtual (Conda) environment and install the project requirements. Follow the instructions to install Conda (miniconda), then start a new virtual environment with the provided ``environment_<your OS>.yml`` file.

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
