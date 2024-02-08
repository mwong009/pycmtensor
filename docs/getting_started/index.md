# Introduction

---

## Why PyCMTensor?

Performing mathematical operations and evaluating models that involve choice-based utility expressions can be challenging and time-consuming. This is especially true when dealing with alternative specific utilities of varying dimensionalities or when specifying neural networks within a utility specification.

Common Python deep learning libraries like TensorFlow, Torch, Keras, or Scikit-learn typically represent entire datasets as products of n-dimensional arrays. However, they often overlook the number of variables used or the specification of independent taste coefficients in each part of the output choice probability equation.
These libraries also tend to obscure the underlying operations that define the inputs of a random utility equation. This makes it difficult to conduct hypothesis testing or run statistical tests without extensive modifications or ad hoc calculations.

While these advanced Python deep learning libraries can be used to evaluate choice models, they are not explicitly designed for estimating choice models, particularly more complex ones like mixed logit. These deep learning models are primarily geared towards model prediction, not choice model estimation where clear interpretation of utility coefficients is crucial.

### What can PyCMTensor do?

PyCMTensor can be used for specifying, estimating, and conducting statistical tests on hybrid discrete choice models. It uses optimized tensor operations through [Aesara](https://aesara.readthedocs.io/en/latest/). 

The library is built for estimating hybrid neural networks, Logit models, and Mixed Logit models. The models in PyCMTensor are built on computational graphs and are estimated using backpropagation algorithms.

## Project goals

PyCMTensor aims to combine the interpretability of choice modelling syntaxes and expressions with the computational efficiency of Aesara tensor libraries. It specializes in defining 'hybrid' utility expressions and simplifies the process of integrating a deep neural network into choice models such as TasteNet or ResLogit. Unlike other deep learning libraries, PyCMTensor focuses on econometric modelling and statistical testing over purely predictive or classification models.

The objectives of this project include:

- Offering a flexible and customizable platform for implementing 'hybrid' neural network discrete choice models.
- Facilitating the development and introduction of deep learning algorithms and methods for choice modelling domain experts and researchers who are already somewhat familiar with neural networks.
- Developing hybrid neural network-based utility specification expressions that are logical and consistent with other conventional choice modelling tools.
- Increasing the computational efficiency of estimating large scale choice models with machine learning algorithms and optimization methods.
- Providing a tool for researchers with similar interests to contribute to an ecosystem for estimating hybrid discrete choice models.

## Key Features

PyCMTensor offers the following features:

- Interpretable and customizable utility specification syntaxes.
- Ability to specify neural networks with weight and bias parameters inside utility functions, such as TasteNet.
- Tools to perform specification testing, analyze covariances, and calculate standard errors for taste parameters.
- Fast execution of model estimation, including simulation-based methods like Mixed Logit models, using a computational graph.
- Model estimation tuning with 1st order methods (e.g., Adam, Stochastic Gradient Descent) or 1.5th order methods (e.g., Stochastic BFGS).

While other choice modelling estimation software in Python are available, such as Biogeme, xlogit, and PyLogit, PyCMTensor aims to fully implement deep learning-based methods in a simplified syntax for utility equation specification.

## Roadmap

PyCMTensor is a work in progress. Several proposed feature implementations need to be completed, and ongoing tasks include code maintenance, documentation writing, and testing. 

The following are proposed major feature implementations:

- Implementation of TasteNet and ResLogit hybrid deep learning choice models.
- Optimization algorithms:
  - ~~1st order estimation (Adam, RMSProp, Rprop. Adagrad)~~
  - ~~Stochastic Quasi Newton NFGS (SQN)~~
  - Momentum-based BFGS
- Variational inference estimation

If you are interested in contributing to the development of PyCMTensor, please get in touch.