# Introduction

---

## Why PyCMTensor?

Writing mathematical operations and evaluating models involving choice based utility  expressions can be difficult and time-consuming, especially with alternative specific  utilities of different dimensionalities are involved or when specifying neural networks  within utility specification. 
Typically, Python deep learning libraries such as TensorFlow, Torch, Keras, or Scikit-learn express entire datasets as products of n-dimensional arrays, without regards to the number of variables used or specification of independent taste coefficients in each part of the output choice probability equation. 
These libraries also do not expose the underlying operations that define the  inputs of a random utility equation, making it inconvenient for hypothesis testing or  running statistical tests without cumbersome modification or ad hoc calculations. 

Although these advanced Python deep learning libraries can be used to evaluate choice 
models, they are not explicitly intended to be used for estimating choice models, 
especially more advanced versions such as mixed logit. These deep learning models are first and foremost for model prediction, and not choice model estimation where clear interpretation of utility coefficients are needed.

### What can PyCMTensor do?
PyCMTensor can be used to fully specify hybrid discrete choice models, estimate and generate statistical tests, using optimized tensor operations via [Aesara](https://aesara.readthedocs.io/en/latest/).
It has a particular focus on estimation of hybrid neural networks and Logit models, as well as on Mixed Logit models.
PyCMTensor models are based on computational graphs and models estimated using generalized backpropagation algorithms.

## Project goals

The goal of PyCMTensor is to combine the easy-to-interpet choice modelling syntaxes and expressions while also implementing some of the mathematical operations and computational speed-up using Aesara tensor libraries. 
PyCMTensor focuses on specifying 'hybrid' utility expressions, and making it easy to define a deep neural network inside a choice model such as TasteNet or ResLogit model specifications. 
The distinction of PyCMTensor from other deep learning libraries is that it focuses on  econometric modelling and statistical testing, rather than focusing purely on models  for prediction or classification.


The higher level goals of this project are:

- Provide a flexible and customizable platform for implementing 'hybrid' neural network discrete choice model
- Facilitating development and introduction of deep learning algorithms and methods for choice modelling domain experts and researchers who are already somewhat familiar with neural networks
- Develop hybrid neural network based utility specification expressions and which are logical and consistent with other conventional choice modelling tools
- Increase the computational efficiency of estimating large scale choice models with machine learning algorithms and optimization methods
- Provide a tool for researchers with similar interests to contribute to an ecosystem for estimating hybrid discrete choice models.

## Key features

Main features:

- Interpretable and customizable utility specification syntaxes
- Ability to specifying neural nets with weight and bias parameters inside a utility functions (e.g. TasteNet)
- Perform specification testing, analyze covariances, standard errors for taste parameters.
- Fast execution of model estimation including of simulation based methods, e.g. Mixed Logit models, using computational graph approach
- Model estimating tuning with 1st order (e.g. Adam, Stochastic Gradient Descent) or 1.5th order methods (e.g. Stochastic BFGS)

While other choice modelling estimation software in Python are available, e.g. Biogeme, xlogit, PyLogit, etc., PyCMTensor strives to fully implement deep learning based methods written in a simplified syntax for utility equation specification.

## Roadmap

PyCMTensor is a work in progress, there are several proposed feature implementations that needs to be done and there are still some code maintenance, documentation writing, and testing to be performed. 

The following are proposed major feature implementations:

  -  Implementation of TasteNet and ResLogit hybrid deep learning choice model
  -  Optimization algorithms:
    - ~~1st order estimation (Adam, RMSProp, Rprop. Adagrad)~~
    - ~~Stochastic Quasi Newton NFGS (SQN)~~
    - Momentum-based BFGS
  - Variational inference estimation

If you are interested in contributing to the development of PyCMTensor, please contact me.