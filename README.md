# PyCMTensor

![](https://img.shields.io/pypi/pyversions/pycmtensor)
[![PyPI version](https://badge.fury.io/py/pycmtensor.svg)](https://badge.fury.io/py/pycmtensor)
[![Tests](https://github.com/mwong009/pycmtensor/actions/workflows/tests.yml/badge.svg)](https://github.com/mwong009/pycmtensor/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/pycmtensor/badge/?version=develop)](https://pycmtensor.readthedocs.io/en/develop/?badge=develop)
[![codecov](https://codecov.io/gh/mwong009/pycmtensor/branch/master/graph/badge.svg?token=LFwgggDyjS)](https://codecov.io/gh/mwong009/pycmtensor)
![Licence](https://img.shields.io/badge/Licence-MIT-blue)

A tensor-based choice modelling Python package with deep learning libraries. 
Read the documentation at <https://pycmtensor.readthedocs.io>

## About PyCMTensor

PyCMTensor is a discrete choice model development platform which is designed with the use of deep learning in mind, enabling users to write more complex models using neural networks.
PyCMTensor is build on [Aesara](https://github.com/aesara-devs/aesara), a tensor library, and uses many features commonly found in deep learning packages such as Tensorflow and Keras.
`Aesara` was chosen as the back end mathematical library because of its hackable, open-source nature.
Users of [Biogeme](https://biogeme.epfl.ch) will be familiar with the syntax of PyCMTensor.

This package allows one to incorporate neural networks into discrete choice models that boosts accuracy of model estimates which still being able to produce all the same statistical analysis found in traditional choice modelling software.

PyCMTensor aims to provide developers and researchers with deep learning tools for econometric and travel behaviour modelling with **reproducible** and **interpretable** results.

### PyCMTensor and Biogeme

PyCMTensor improves on Biogeme in situations where much more complex models are necessary, for example, integrating neural networks into discrete choice models.
PyCMTensor also include the ability to estimate models using 1st order stochastic gradient descent methods by default, such as Nesterov Accelerated Gradient, Adam, or RMSProp.
## Features

* Estimate complex choice models with neural networks using deep learning algorithms
* Combines traditional econometric models (e.g. Multinomial Logit) with deep learning models (e.g. ResNets)
* Shares similar programming syntax with `Biogeme`, allowing easy transition between `Biogeme` and PyCMTensor models
* Uses tensor based mathematical operations from the advanced features found in the `Aesara` library

---

## Pre-install

To install PyCMTensor, you need [Conda](https://docs.conda.io/en/latest/miniconda.html) (Full Anaconda works fine, but **miniconda** is recommmended for a minimal installation). 
Ensure that Conda is using at least `Python 3.9`.

Once Conda is installed, install the required dependencies from conda by running the following 
command in your terminal:

**Windows**

```console
conda install mkl-service conda-forge::cxx-compiler conda-forge::m2w64-toolchain -y
```
**Linux**

```console
conda install mkl-service conda-forge::cxx-compiler
```

**Mac OSX**

```console
conda install mkl-service Clang
```

### Install PyCMTensor

Then, run this command in your terminal to download and install the latest branch of PyCMTensor from [PyPI](https://pypi.org/project/pycmtensor/):
```console
pip install pycmtensor -U
```

*Optional*: If you want the development version from the Github repository:
```console
pip install git+https://github.com/mwong009/pycmtensor.git@develop -U
```

The development branch is the most recent update of PyCMTensor. If you want a stable branch (master), remove ``@develop`` at the end of the ``.git`` url.

## Usage

PyCMTensor uses syntax very similar to `Biogeme`. Users of `Biogeme` should be familiar 
with the syntax.

Start an interactive session (`IPython` or Jupyter Notebook) and import:
```Python
import pycmtensor as cmt
```

Several submodules are also important to include:
```Python
from pycmtensor.expressions import Beta # Beta class for model parameters
from pycmtensor.models import MNLogit   # model library
from pycmtensor.optimizers import Adam  # Optimizers
from pycmtensor.results import Results  # for generating results
```

For a full list of submodules and description, refer to [API Reference](/autoapi/index)

## Development

To set up `PyCMTensor` in a local development environment, you need to set up a virtual environment and install the project requirements. Follow the instructions to install Conda (miniconda), then start a new virtual environment with the provided environment_\<your OS\>.yml file.

For example in windows:
```console
conda env create -f environment_windows.yml
```

Next, activate the virtual environment and install poetry via `pip`.
```console
conda activate pycmtensor-dev
pip install poetry
```

Lastly, install the project and development dependencies
```console
poetry install -E dev
```

The virtual environment needs to be activated and commits are done from the virtural environment.

### Simple example: Swissmetro dataset

Using the [swissmetro dataset](https://biogeme.epfl.ch/data.html) from Biogeme, we define a simple MNL model. 

Note:The following is a replication of the results from Biogeme using the `Adam` optimization algorithm.


1. Import the dataset and perform some data santiation
	```Python
	swissmetro = pd.read_csv("data/swissmetro.dat", sep="\t")
	db = cmt.Database(name="swissmetro", pandasDatabase=swissmetro, choiceVar="CHOICE")
	globals().update(db.variables)
	# Removing some observations
	db.data.drop(db.data[db.data["CHOICE"] == 0].index, inplace=True)
	db.data["CHOICE"] -= 1  # set the first choice index to 0
	db.choices = [0, 1, 2]
	db.autoscale(
		variables=["TRAIN_CO", "TRAIN_TT", "CAR_CO", "CAR_TT", "SM_CO", "SM_TT"],
		default=100.0,
		verbose=False,
	)
	```

	``cmt.Database()`` loads the dataset and defines tensor variables automatically.

2. Initialize the model parameters and specify the utility functions and availability conditions
	```Python
	b_cost = Beta("b_cost", 0.0, None, None, 0)
	b_time = Beta("b_time", 0.0, None, None, 0)
	asc_train = Beta("asc_train", 0.0, None, None, 0)
	asc_car = Beta("asc_car", 0.0, None, None, 0)
	asc_sm = Beta("asc_sm", 0.0, None, None, 1)

	U_1 = b_cost * db["TRAIN_CO"] + b_time * db["TRAIN_TT"] + asc_train
	U_2 = b_cost * db["SM_CO"] + b_time * db["SM_TT"] + asc_sm
	U_3 = b_cost * db["CAR_CO"] + b_time * db["CAR_TT"] + asc_car
	``` 

3. Define the Multinomial Logit model
	```Python
	mymodel = MNLogit(u=U, av=AV, database=db, name="Multinomial Logit")
	mymodel.add_params(locals()) # load Betas into the model
	```

5. (optional) Define the model hyperparameters
	```Python
	mymodel.config["patience"] = 9000
	mymodel.config["max_epoch"] = 500
	mymodel.config["base_lr"] = 0.0012
	mymodel.config["max_lr"] = 0.002
	mymodel.config["learning_scheduler"] = "ConstantLR"
	```

6. Call the training function and save the trained model
	```Python
	model = cmt.train(model=mymodel, database=db, optimizer=Adam)  # we use the Adam Optimizer
	```

7. Generate the statistics and correlation matrices
	```Python
	results = Results(model, db, prnt=False)
	print(results)
	results.generate_beta_statistics()
	results.print_beta_statistics()
	results.print_correlation_matrix()

	```

	Sample output: 

		Python 3.10.4 | packaged by conda-forge | (main, Mar 30 2022, 08:38:02) [MSC v.1916 64 bit (AMD64)]
		[2022-08-12 18:51:40] INFO: Building model...
		[2022-08-12 18:51:52] INFO: Training model...
		[2022-08-12 18:51:55] INFO: Maximum iterations reached. Terminating...
		[2022-08-12 18:51:55] INFO: Optimization complete with accuracy of 61.937%.
		[2022-08-12 18:51:55] INFO: Max log likelihood reached @ epoch 57.

		Results
		------
		Model: Multinomial Logit
		Build time: 00:00:12
		Estimation time: 00:00:03
		Estimation rate: 3400.838 iter/s
		Seed value: 7577
		Number of Beta parameters: 4
		Sample size: 10719
		Excluded data: None
		Init loglikelihood: -11093.627
		Final loglikelihood: -9165.567
		Final loglikelihood reached at: epoch 57
		Likelihood ratio test: 3856.120
		Accuracy: 61.937%
		Rho square: 0.174
		Rho bar square: 0.173
		Akaike Information Criterion: 18339.13
		Bayesian Information Criterion: 18368.25
		Final gradient norm: 0.121

		Model statistics:
		              Value   Std err     t-test   p-value Rob. Std err Rob. t-test Rob. p-value
		asc_car    0.013287  0.030614   0.434002  0.664287     0.159125    0.083498     0.933456
		asc_train -0.537674  0.037544 -14.321085       0.0     0.014821  -36.278684          0.0
		b_cost     0.021882  0.002227   9.824814       0.0     0.005462     4.00618     0.000062
		b_time    -0.604866  0.035116 -17.224787       0.0     0.514255   -1.176199     0.239515

		Correlation matrix:
		             b_cost    b_time  asc_train   asc_car
		b_cost     1.000000 -0.092697   0.171935  0.269662
		b_time    -0.092697  1.000000  -0.710780 -0.596636
		asc_train  0.171935 -0.710780   1.000000  0.603376
		asc_car    0.269662 -0.596636   0.603376  1.000000

8. Plot the training performance and accuracy

	![](https://github.com/mwong009/pycmtensor/blob/master/docs/_static/viz/fig.png)

9. Compute the elasticities

	![](https://github.com/mwong009/pycmtensor/blob/master/docs/_static/viz/els.png)

10. Visualize the computation graph
	```Python
	import aesara.d3viz as d3v
	from aesara import printing
	printing.pydotprint(mymodel.cost, "graph.png")
	```

	![](https://github.com/mwong009/pycmtensor/blob/master/docs/_static/viz/print.png)

---

## Credits

PyCMTensor was inspired by [Biogeme](https://biogeme.epfl.ch) and aims to provide deep learning modelling tools for transport modellers and researchers.

This package template was generated with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.
