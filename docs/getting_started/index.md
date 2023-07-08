# Introduction

---

## Why PyCMTensor?

Writing mathematical operations and evaluating models involving choice based utility 
expressions can be difficult and time-consuming, especially with alternative specific 
utilities of different dimensionalities are involved or when specifying neural networks 
within utility specification. Typically, Python deep learning libraries such as 
TensorFlow, Torch, Keras, or Scikit-learn express entire datasets as products of 
n-dimensional arrays, without regards to the number of variables used or specification 
of independent taste coefficients in each part of the output choice probability 
equation. These libraries also do not expose the underlying operations that define the 
inputs of a random utility equation, making it inconvenient for hypothesis testing or 
running statistical tests without cumbersome modification or ad hoc calculations. 

Although these advanced Python deep learning libraries can be used to evaluate choice 
models, they are not explicitly intended to be used for estimating choice models, 
especially more advanced versions such as mixed logit. These deep learning models are first and foremost for model prediction, and not choice model estimation where clear interpretation of utility coefficients are needed.

### What can PyCMTensor do?
Currently, PyCMTensor can be used to fully specify Multinomial Logit and Mixed Logit models, estimate and generate statistical tests, using optimized tensor operations via Aesara tensor libraries.

## Project goals

/// admonition
The goal of PyCMTensor is to combine the easy-to-interpet choice modelling syntaxes and 
expressions (e.g. Biogeme) while also implementing some of the tensor-based operations 
and computational speed-up of a deep learning library through Aesara. PyCMTensor 
focuses on specifying utility expressions, and making it easy to define a deep neural 
network inside a choice model such as TasteNet or ResLogit model specifications. The 
distinction of PyCMTensor from other deep learning libraries is that it focuses on 
econometric modelling and statistical testing, rather than focusing purely on models 
for prediction.
///

The higher level goals of this project are:

- provide tools to implement "hybrid" deep learning based discrete choice model
- facilitating development of deep learning tool for domain experts and researchers who are already familiar with discrete choice modelling with Biogeme
- Using the library to demonstrate innovative deep learning methods applied to econometric modelling
- make utility specification and prototyping at the meta-level more logical and consistent with choice modelling equations
- increase computational efficiency of estimating large scale choice models with machine learning techniques
- provide a platform for researchers with similar interests to contribute to an ecosystem for estimating advanced discrete choice models.

## Key features and differences

Main features:

- Utility specification syntax writing in Python.
- Perform statistical tests and  generate var-covar matrices for taste parameters. 
- Fast execution of model estimation including of simulation based methods, e.g. Mixed Logit models.
- Model estimation with 1st order (Stochastic GD) or 2nd order methods (BFGS).
- Specifying neural nets with weight and bias parameters inside a utility function. TODO

While other choice modelling estimation software are available, e.g. ..., PyCMTensor strives to fully implement deep learning based methods written in the same syntax format as Biogeme. Different software programs may occasionally vary in their behaviour and estimation results. The following are some of the key differences between PyCMTensor and other choice modelling estimation packages:

## Roadmap

PyCMTensor is a work in progress, there are several proposed feature implementations that needs to be done and there are still some code maintenance, documentation writing, and testing to be performed. 

The following are proposed major feature implementations:

  - Implementation of TasteNet and ResLogit hybrid deep learning choice models
  - Optimization algorithms:
    - Stochastic Newton Method (SNM)
    - Momentum-based BFGS
  - Variational inference estimation

If you are interested in contributing to the development of PyCMTensor, please contact me.