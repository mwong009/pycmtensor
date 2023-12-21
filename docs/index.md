![PyCMTensor](assets/img/logo.jpg)

[![Licence](https://img.shields.io/badge/Licence-MIT-blue)](about/licence.md)
![](https://img.shields.io/pypi/pyversions/pycmtensor) 
[![PyPI version](https://badge.fury.io/py/pycmtensor.svg)](https://badge.fury.io/py/pycmtensor) 
[![codecov](https://codecov.io/gh/mwong009/pycmtensor/branch/master/graph/badge.svg?token=LFwgggDyjS)](https://codecov.io/gh/mwong009/pycmtensor) 
[![Downloads](https://static.pepy.tech/personalized-badge/pycmtensor?period=month&units=international_system&left_color=grey&right_color=orange&left_text=downloads/month)](https://pepy.tech/project/pycmtensor) 
[![DOI](https://zenodo.org/badge/460802394.svg)](https://zenodo.org/badge/latestdoi/460802394)

A Tensor-based choice modelling estimation package written in Python

# Welcome

PyCMTensor is a Python library that use tensor-based computations for discrete choice modelling and estimation. It specializes in the estimation of hybrid neural networks, Logit models, and Mixed Logit models. The models in PyCMTensor are built on computational graphs and are estimated using generalized backpropagation algorithms.

This library provides the tools to fully specify and estimate Multinomial Logit and Mixed Logit models. It also enables the generation of statistical test results for comprehensive econometric analysis. With PyCMTensor, you can perform model estimation using computational graphs, making it a powerful tool for advanced statistical modelling.

## Key features

**Model specification**

PyCMTensor provides the ability to define and customize alternative-specific linear and non-linear utility functions, random variables, and deep neural networks. It maintains a record of model taste parameters, gradients, and Hessian matrices, facilitating econometric interpretability, specification testing, prediction, and elasticity analysis.

**Cost functions**

PyCMTensor offers a variety of cost functions to assess model performance, including negative log-likelihood, mean squared error, and KL divergence. It also allows for the evaluation of model accuracy using out-of-sample prediction probabilities or discrete choices (Argmax).

**Data processing**

PyCMTensor simplifies the process of segmenting datasets for training and validation, allowing for a focused approach to model development. It supports arithmetic operations and boolean expressions on variables and parameters within the utility function, eliminating the need for pre-processing the data file. Additionally, it enables minibatch training for faster estimation on large datasets.

**Model tuning**

PyCMTensor incorporates a collection of 1st order and 1.5th order optimization routines and learning rate schedulers, designed specifically for estimating choice models. This feature aids in fine-tuning the model for optimal performance.


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

