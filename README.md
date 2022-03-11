# PyCMTensor

![](https://img.shields.io/badge/version-0.6.2-orange)
[![Documentation Status](https://readthedocs.org/projects/pycmtensor/badge/?version=latest)](https://pycmtensor.readthedocs.io/en/latest/?version=latest)
[![](https://pyup.io/repos/github/mwong009/pycmtensor/shield.svg)](https://pyup.io/repos/github/mwong009/pycmtensor)
![Licence](https://img.shields.io/badge/Licence-MIT-blue)

A tensor-based choice modelling Python package with deep learning capabilities

`PyCMTensor` is a discrete choice model development platform which is designed with the use of deep learning in mind, enabling users to write more complex models using neural networks.
`PyCMTensor` is build on [Aesara library](https://github.com/aesara-devs/aesara), and uses many features commonly found in deep learning packages such as Tensorflow and Keras.
`Aesara` was chosen as the back end mathematical library because of its hackable, open-source nature.
As users of [Biogeme](https://biogeme.epfl.ch), you will be familiar with the syntax of `PyCMTensor` and as it is built on top of existing `Biogeme` choice models.

The combination of `Biogeme` and `Aesara` allows one to incorporate neural networks into discrete choice models that boosts accuracy of model estimates which still being able to produce all the same statistical analysis found in traditional choice modelling software.


<!-- ![](https://img.shields.io/pypi/v/pycmtensor.svg) -->


## Features

* Efficiently estimate complex choice models with neural networks using deep learning algorithms
* Combines traditional econometric models (Multinomial Logit) with deep learning models (ResNets)
* Similar programming syntax as `Biogeme`, allowing easy substitution between `Biogeme` and `PyCMTensor` methods
* Uses tensor based mathematical operations from the advanced features found in the `Aesara` library

## Install

To install PyCMTensor, you need [Conda](https://docs.conda.io/en/latest/miniconda.html) (Full Anaconda works fine, but **miniconda** is recommmended for a minimal installation)

Once Conda is installed, install the required dependencies from conda by running the following 
command in your terminal:

```console
$ conda install pip git cxx-compiler m2w64-toolchain libblas libpython mkl numpy
```

>Note: Mac OSX user should also install `Clang` for a fast compiled code.

Then, run this command in your terminal to download and install the development branch of `PyCMTensor`:

```console
$ pip install git+https://github.com/mwong009/pycmtensor.git@develop -U
```

The development branch is the most up-to-date version of `PyCMTensor`. If you want a stable branch, remove ``@develop`` at the end of the url.

## How to use

PyCMTensor uses syntax very similar to `Biogeme`. Users of `Biogeme` should be familiar 
with the syntax.

Start an interactive session (IPython or Jupyter Notebook) and import PyCMTensor:
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

## Simple example: Swissmetro dataset

Using the swissmetro dataset from Biogeme to define a simple MNL model. 

The following is a replication of the results from Biogeme using the `Adam` optimization algorithm and a `Cyclic learning rate`. For further examples including the ResLogit model, refer **here**.

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
		verbose=False
	) # automatically scales features by 1/100.
	```

	``cmt.Database()`` loads the dataset and automatically defines symbolic Tensor Variables.

2. Initialize the model parameters
	```Python
	b_cost = Beta("b_cost", 0.0, None, None, 0)
	b_time = Beta("b_time", 0.0, None, None, 0)
	asc_train = Beta("asc_train", 0.0, None, None, 0)
	asc_car = Beta("asc_car", 0.0, None, None, 0)
	asc_sm = Beta("asc_sm", 0.0, None, None, 1)
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
	mymodel.add_params(locals())
	```

5. Set up the training hyperparameters
	```Python
	mymodel.config["patience"] = 20000
	mymodel.config["base_lr"] = 0.0012
	mymodel.config["max_lr"] = 0.002
	mymodel.config["learning_scheduler"] = "CyclicLR"
	mymodel.config["cyclic_lr_step_size"] = 8
	mymodel.config["cyclic_lr_mode"] = "triangular2"
	```

6. Call the training function and save the trained model
	```Python
	model = cmt.train(mymodel, database=db, optimizer=Adam, batch_size=128, 
	                  max_epoch=999)
	```

7. Generate the statistics and correlation matrices
	```Python
	result = Results(model, db, show_weights=True)
	result.print_beta_statistics()
	result.print_correlation_matrix()
	```

8. Plot the training performance and accuracy
	![](../viz/fig.png)

8. Visualize the computation graph
	```Python
	import aesara.d3viz as d3v
	from aesara import printing
	printing.pydotprint(mymodel.cost, "graph.png")
	```
	![](../viz/print.png)


## Credits

PyCMTensor was inspired by [Biogeme](https://biogeme.epfl.ch) and aims to provide deep learning modelling tools for transport modellers and researchers.

This package was generated with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.