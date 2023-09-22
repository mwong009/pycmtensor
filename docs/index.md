![PyCMTensor](assets/img/logo.jpg)

[![Licence](https://img.shields.io/badge/Licence-MIT-blue)](about/licence.md)
![](https://img.shields.io/pypi/pyversions/pycmtensor) 
[![PyPI version](https://badge.fury.io/py/pycmtensor.svg)](https://badge.fury.io/py/pycmtensor) 
[![codecov](https://codecov.io/gh/mwong009/pycmtensor/branch/master/graph/badge.svg?token=LFwgggDyjS)](https://codecov.io/gh/mwong009/pycmtensor) 
[![Downloads](https://static.pepy.tech/personalized-badge/pycmtensor?period=month&units=international_system&left_color=grey&right_color=orange&left_text=downloads/month)](https://pepy.tech/project/pycmtensor) 
[![DOI](https://zenodo.org/badge/460802394.svg)](https://zenodo.org/badge/latestdoi/460802394)

A Tensor-based choice modelling estimation package written in Python

# Welcome

PyCMTensor is a tensor-based discrete choice modelling estimation Python library package.
It has a particular focus on estimation of hybrid neural networks and Logit models, as well as on Mixed Logit models. PyCMTensor models are based on computational graphs and models estimated using generalized backpropagation algorithms.
PyCMTensor can be used to fully specify Multinomial Logit and Mixed Logit models, perform model estimation using computational graphs and generate statistical test results for econometric analysis.

## Key features

**Model specification**

PyCMTensor can be used to specify and customize alternative specific linear and non-linear utility functions, random variables and deep neural networks.
PyCMTensor can keep track of model taste parameters, gradients and hessian matrices for econometric interpretability, specification testing, prediction, and elasticity analysis.

**Cost functions**

To evaluate model performance, various cost functions are available, which include (negative) log likelihood, mean squared error, and KL divergence. Model accuracy can also be evaluated using out-of-sample prediction probabilities or discrete choices (Argmax). 

**Data processing**

Datasets can be split into different segments for training and validation easily, to prioritize different aspects of the model. Arithmetic operations and boolean expressions can be used on variables and parameters within the utility function without pre-processing the datafile. Minibatch training is possible for estimation speedup on large datasets.

**Model tuning**

PyCMTensor includes a set of 1st order and 1.5th order optimization routines and learning rate schedulers for estimating choice models. 

## Getting started

* [**Introduction**](getting_started/index.md) - A brief introduction of the project and its features
* [**Installing PyCMTensor**](getting_started/installation.md) - Instructions to install PyCMTensor
* [**Overview**](getting_started/overview.md) - A short 5-minute quick start to estimating your first model
* [**Troubleshooting and tips**](getting_started/troubleshooting.md) - Some tips for common problems and fixes

## Examples

Some basic working code examples

## User guide

* [**User guide**](user_guide/index.md) - Detailed guide on using PyCMTensor
* [**PyCMTensor configuration**](user_guide/configuration.md) - How to modify PyCMTensor attributes

## Developer guide

* [**Developer guide**](developer_guide/index.md) - Guide for developers
* [**API reference**](developer_guide/api/index.md)

## About

* [**Contributing**](about/contributing.md)
* [**Release notes**](about/release_notes.md)
* [**Licence**](about/licence.md)
* [**Citation**](about/citation.md)

