# PyCMTensor

![](https://img.shields.io/pypi/pyversions/pycmtensor)
[![PyPI version](https://badge.fury.io/py/pycmtensor.svg)](https://badge.fury.io/py/pycmtensor)
[![Tests](https://github.com/mwong009/pycmtensor/actions/workflows/tests.yml/badge.svg)](https://github.com/mwong009/pycmtensor/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/pycmtensor/badge/?version=develop)](https://pycmtensor.readthedocs.io/en/develop/?badge=develop)
[![codecov](https://codecov.io/gh/mwong009/pycmtensor/branch/master/graph/badge.svg?token=LFwgggDyjS)](https://codecov.io/gh/mwong009/pycmtensor)
![Licence](https://img.shields.io/badge/Licence-MIT-blue)

A tensor-based choice modelling Python package with deep learning capabilities. 
Read the documentation at <https://pycmtensor.readthedocs.io>

## About PyCMTensor

PyCMTensor is a discrete choice model development platform which is designed with the use of deep learning in mind, enabling users to write more complex models using neural networks.
PyCMTensor is build on [Aesara library](https://github.com/aesara-devs/aesara), and uses many features commonly found in deep learning packages such as Tensorflow and Keras.
`Aesara` was chosen as the back end mathematical library because of its hackable, open-source nature.
As users of [Biogeme](https://biogeme.epfl.ch), you will be familiar with the syntax of PyCMTensor and as it is built on top of existing `Biogeme` choice models.

The combination of `Biogeme` and `Aesara` allows one to incorporate neural networks into discrete choice models that boosts accuracy of model estimates which still being able to produce all the same statistical analysis found in traditional choice modelling software.

PyCMTensor aims to provide developers and researchers with deep learning tools for econometric modelling and travel behaviour modell with **reproducible** and **interpretable** results.

### PyCMTensor and Biogeme

PyCMTensor is meant to complement Biogeme where much more complex models are necessary, for example, integrating neural networks into discrete choice models.
PyCMTensor also include the ability to estimate models using 1st order stochastic gradient descent methods by default, such as Nesterov Accelerated Gradient, Adam, or RMSProp.
## Features

* Estimate complex choice models with neural networks using deep learning algorithms
* Combines traditional econometric models (e.g. Multinomial Logit) with deep learning models (ResNets)
* Shares similar programming syntax with `Biogeme`, allowing easy transition between `Biogeme` and PyCMTensor methods
* Uses tensor based mathematical operations from the advanced features found in the `Aesara` library

---

## Pre-install

To install PyCMTensor, you need [Conda](https://docs.conda.io/en/latest/miniconda.html) (Full Anaconda works fine, but **miniconda** is recommmended for a minimal installation). 
Ensure that Conda is using at least `Python 3.9`.

Once Conda is installed, install the required dependencies from conda by running the following 
command in your terminal:

**Windows**

```console
conda install pip git conda-forge::cxx-compiler conda-forge::libpython blas mkl-service numpy
```
**Linux/MacOS**

```console
conda install blas mkl-service conda-forge::cxx-compiler
```

Note: MacOS user should also [install](https://www.ics.uci.edu/~pattis/common/handouts/macclion/clang.html) `Clang` for a fast compiled code.

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

---

## How to use

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

### Simple example: Swissmetro dataset

Using the [swissmetro dataset](https://biogeme.epfl.ch/data.html) from Biogeme, we define a simple MNL model. 

Note:The following is a replication of the results from Biogeme using the `Adam` optimization algorithm. For further examples including the ResLogit model, refer **here**.


1. Import the dataset and perform some data santiation
	```Python
	swissmetro = pd.read_csv("swissmetro.dat", sep="\t")
	db = cmt.Database(name="swissmetro", pandasDatabase=swissmetro, choiceVar="CHOICE")
	globals().update(db.variables)
	# additional steps to format database
	db.data["CHOICE"] -= 1 # set the first choice to 0
	db.choices = sorted(db.data["CHOICE"].unique()) # save original choices
	db.autoscale(
		variables=['TRAIN_CO', 'TRAIN_TT', 'CAR_CO', 'CAR_TT', 'SM_CO', 'SM_TT'], 
		default=100., 
	) # automatically scales features by 1/100.
	```

	``cmt.Database()`` loads the dataset and defines Tensor Variables automatically.

2. Initialize the model parameters
	```Python
	b_cost = Beta("b_cost", 0.0, None, None, 0)
	b_time = Beta("b_time", 0.0, None, None, 0)
	asc_train = Beta("asc_train", 0.0, None, None, 0)
	asc_car = Beta("asc_car", 0.0, None, None, 0)
	asc_sm = Beta("asc_sm", 0.0, None, None, 1)  # set to 1 to keep it fixed
	``` 

3. Specify the utility functions and availability conditions
	```Python
	U_1 = b_cost * db["TRAIN_CO"] + b_time * db["TRAIN_TT"] + asc_train
	U_2 = b_cost * db["SM_CO"] + b_time * db["SM_TT"] + asc_sm
	U_3 = b_cost * db["CAR_CO"] + b_time * db["CAR_TT"] + asc_car
	U = [U_1, U_2, U_3]
	AV = [db["TRAIN_AV"], db["SM_AV"], db["CAR_AV"]]
	```

4. Specify the model ``MNLogit``
	```Python
	mymodel = MNLogit(u=U, av=AV, database=db, name="mymodel")
	mymodel.add_params(locals())  # load Betas into the model
	```

5. Define the training hyperparameters
	```Python
	mymodel.config["patience"] = 20000
	mymodel.config["max_epoch"] = 499
	mymodel.config["base_lr"] = 0.0012
	mymodel.config["max_lr"] = 0.002
	mymodel.config["learning_scheduler"] = "CyclicLR"
	mymodel.config["cyclic_lr_step_size"] = 8
	mymodel.config["cyclic_lr_mode"] = "triangular2"
	```

6. Call the training function and save the trained model
	```Python
	model = cmt.train(mymodel, database=db, optimizer=Adam)  # we use the Adam Optimizer
	```

7. Generate the statistics and correlation matrices
	```Python
	result = Results(model, db, show_weights=True)
	result.print_beta_statistics()
	result.print_correlation_matrix()
	```

	Sample output: 

		Optimization complete with accuracy of 62.618%. Max loglikelihood reached @ epoch 195.	

		Results for model: mymodel
		Build time: 00:00:13
		Estimation time: 00:00:28
		Estimation rate: 13.781 epochs/s
		Seed value: 999
		Number of Beta parameters: 4
		Sample size: 6768
		Excluded data: 3960
		Init loglikelihood: -6964.663
		Final loglikelihood: -5590.672
		Final loglikelihood reached at: epoch 195
		Likelihood ratio test: 2747.982
		Accuracy: 62.618%
		Rho square: 0.197
		Rho bar square: 0.197
		Akaike Information Criterion: 11189.34
		Bayesian Information Criterion: 11216.62
		Final gradient norm: 0.111

		Statistical Analysis:
		              Value   Std err     t-test   p-value Rob. Std err Rob. t-test Rob. p-value
		asc_car    0.111877  0.042071   2.659267  0.007831     0.038512    2.905005     0.003672
		asc_train -0.624174   0.05471 -11.408845       0.0     0.014402   -43.33787          0.0
		b_cost     0.002601  0.002547    1.02136  0.307084     0.003616    0.719242     0.471992
		b_time     -1.16109  0.054086 -21.467576       0.0     0.005372 -216.155293          0.0

		Correlation matrix:
		             b_cost    b_time  asc_train   asc_car
		b_cost     1.000000 -0.105761   0.154368  0.283711
		b_time    -0.105761  1.000000  -0.724388 -0.659056
		asc_train  0.154368 -0.724388   1.000000  0.606882
		asc_car    0.283711 -0.659056   0.606882  1.000000

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

This package was generated with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.
