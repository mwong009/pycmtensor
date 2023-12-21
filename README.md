# PyCMTensor

![PyCMTensor](docs/assets/img/logo.jpg)

[![Licence](https://img.shields.io/badge/Licence-MIT-blue)](about/licence.md)
![](https://img.shields.io/pypi/pyversions/pycmtensor) 
[![PyPI version](https://badge.fury.io/py/pycmtensor.svg)](https://badge.fury.io/py/pycmtensor) 
[![codecov](https://codecov.io/gh/mwong009/pycmtensor/branch/master/graph/badge.svg?token=LFwgggDyjS)](https://codecov.io/gh/mwong009/pycmtensor) 
[![Downloads](https://static.pepy.tech/personalized-badge/pycmtensor?period=month&units=international_system&left_color=grey&right_color=orange&left_text=downloads/month)](https://pepy.tech/project/pycmtensor) 
[![DOI](https://zenodo.org/badge/460802394.svg)](https://zenodo.org/badge/latestdoi/460802394)

*Advanced Tensor-based Choice Modelling Python package*

[Introduction](#introduction) | [Install](#installation) | [Key Features](#key-features)

---

## Introduction

PyCMTensor is a Python library that use tensor-based computations for discrete choice modelling and estimation. It specializes in the estimation of hybrid neural networks, Logit models, and Mixed Logit models. The models in PyCMTensor are built on computational graphs and are estimated using generalized backpropagation algorithms.

This library provides the tools to fully specify and estimate Multinomial Logit and Mixed Logit models. It also enables the generation of statistical test results for comprehensive econometric analysis. With PyCMTensor, you can perform model estimation using computational graphs, making it a powerful tool for advanced statistical modelling.

## Installation

**PyCMTensor** is available on Conda-forge:

```bash
conda install -c conda-forge pycmtensor
```

## Quick start

A simple Multinomial logit model. Dataset: [Swissmetro](http://transp-or.epfl.ch/data/swissmetro.dat)

```python
import pandas as pd
from pycmtensor.dataset import Dataset
from pycmtensor.expressions import Beta
from pycmtensor.models import MNL, train
import pycmtensor.optimizers as optim

def mnl_model(filename='http://transp-or.epfl.ch/data/swissmetro.dat'):
	# Load the CSV file into a DataFrame
	df = pd.read_csv(filename, sep='\t')

	# Load the DataFrame into a Dataset object
	ds = Dataset(df, choice="CHOICE")
	ds.scale_variable("TRAIN_TT", 100)
	ds.scale_variable("SM_TT", 100)
	ds.scale_variable("CAR_TT", 100)
	ds.scale_variable("TRAIN_CO", 100)
	ds.scale_variable("SM_CO", 100)
	ds.scale_variable("CAR_CO", 100)
	ds.split(0.8)

	# Define the alternative specific constants (ASCs) for each mode of transport
    ASC_TRAIN = Beta("ASC_TRAIN", 0., None, None, 0)
	ASC_SM = Beta("ASC_SM", 0., None, None, 0)
	ASC_CAR = Beta("ASC_CAR", 0., None, None, 1)
	B_COST = Beta("B_COST", -1., None, None, 1)
	B_TIME_TRAIN = Beta("B_TIME_TRAIN", 0., None, None, 0)
	B_TIME_SM = Beta("B_TIME_SM", 0., None, None, 0)
	B_TIME_CAR = Beta("B_TIME_CAR", 0., None, None, 0)
	B_SEAT = Beta("B_SEAT", 0., None, None, 0)

	# Define the utility functions for each mode of transport
	V_TRAIN = ASC_TRAIN + B_TIME_TRAIN * ds["TRAIN_TT"] + B_COST * ds["TRAIN_CO"]
	V_SM = ASC_SM + B_TIME_SM * ds["SM_TT"] + B_COST * ds["SM_CO"] + B_SEAT * ds["SM_SEATS"]
	V_CAR = ASC_CAR + B_TIME_CAR * ds["CAR_TT"] + B_COST * ds["CAR_CO"]

	# Define the model
	U = [V_TRAIN, V_SM, V_CAR]
	AV = [ds["TRAIN_AV"], ds["SM_AV"], ds["CAR_AV"]]
	model = MNL(ds, locals(), utility=U, av=AV)
	
	return model, ds

# Train model
model, ds = mnl_model()
train(model, ds, optimizer=optim.Adam, max_epochs=2000, base_learning_rate=0.1, convergence_threshold=1e-3)

# Print model results
print(model.results.beta_statistics())
print(model.results.model_statistics())
print(model.results.benchmark())
```

## Key Features

- **Interpretable and customizable utility specification syntaxes**: Easily define your models with an intuitive syntax.
- **Neural network specification**: Specify neural networks with weight and bias parameters inside utility functions (e.g., TasteNet).
- **Comprehensive analysis tools**: Perform specification testing, analyze covariances, and compute standard errors for taste parameters.
- **Fast model estimation**: Quickly estimate models, including simulation-based methods like Mixed Logit models, using a computational graph approach.
- **Flexible optimization methods**: Tune the model estimation with 1st order methods (e.g., Adam, Stochastic Gradient Descent) or 1.5th order methods (e.g., Stochastic BFGS).

While other choice modelling estimation software in Python are available, such as Biogeme, xlogit, and PyLogit, PyCMTensor sets itself apart by fully implementing deep learning-based methods with a simplified syntax for utility equation specification.

## Documentation

For more information on how to use PyCMTensor, please refer to our [documentation](link-to-documentation).