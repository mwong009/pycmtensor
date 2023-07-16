# PyCMTensor Configuration

---

## Guide

The [`config`](../developer_guide/api/config.md#pycmtensor.config.Config) module contains `attributes` that is used for setting the model training hyperparameters, type of optimizer to use, random seed value, and other customizable values. These attributes are loaded when importing the [`pycmtensor`](../developer_guide/api/__init__.md#pycmtensor) module, but can be modified at any time before invoking the [`train()`]() method.

Display the list of configuration settings with the following in Python:
```python
import pycmtensor
print(pycmtensor.config)
```

Set or update a given configuration with the following:
```python
pycmtensor.config.add('seed', 100)
```

## Configuration attributes

### `config.seed`

:	Seed value for random number generators. 

	Default: `100`

### `config.batch_size`

:	Number of samples processed on each iteration of the model update

	Default: `32`

### `config.max_epochs`

:	Maximum number of model update epochs

	Default: `500`

### `config.patience`

:	Process this number of iterations at minimum

	Default: `2000`

### `config.patience_increase`

:	Increase patience by this factor if model does not converge

	Default: `2`

### `config.validation_threshold`

:	The factor of the validation error score to meet in order to register an improvement

	Default: `1.003`

### `config.convergence_threshold`

:	The gradient norm convergence threshold before model termination

	Default: `1e-4`

### `config.base_learning_rate`

:	The initial learning rate

	Default: `1.003`

### `config.max_learning_rate`

:	The maximum learning rate (additional option for various schedulers)

	Default: `0.1`

### `config.min_learning_rate`

:	The minimum learning rate (additional option for various schedulers)

	Default: `1e-5`

### `config.optimizer`

:	Optimization algorithm to use for model estimation

	Default: `pycmtensor.optimizers.Adam`

	Possible options are: 
	
	- 1st order optimizers: `Adam`, `Nadam`, `Adam`, `Adamax`, `Adadelta`, `RMSProp`, `Momentum`, `NAG`, `AdaGrad`, `SGD`
  	- 2nd order optimizers: `BFGS`

	!!! note
		`config.optimizer` takes a [`pycmtensor.optimizers.Optimizer`]() class object as a value. Refer to [here]() for more information on optimizers.

### `config.BFGS_warmup`

:	Discards this number of hessian matrix updates when running the `BFGS` algorithm

	Default: `10`

### `config.lr_scheduler`

:	Learning rate scheduler to use for model estimation

	Default: `pycmtensor.scheduler.ConstantLR`

	Possible options are: 
	
	- `ConstantLR`, `StepLR`, `PolynomialLR`, `CyclicLR`, `TriangularCLR`, `ExpRangeCLR`

	!!! note
		`config.lr_scheduler` takes a [`pycmtensor.optimizers.Scheduler`]() class object as a value. Refer to [here]() for more information on learning rate scheduler.

### `config.lr_ExpRangeCLR_gamma`

:	Gamma parameter for `ExpRangeCLR`

	Default: `0.5`

### `config.lr_stepLR_factor`

:	Drop step multiplier factor for `stepLR`

	Default: `0.5`

### `config.lr_stepLR_drop_every`

:	Drop learning rate every n steps for `stepLR`

	Default: `10`

### `config.lr_CLR_cycle_steps`

:	Steps per cycle for `CyclicLR`

	Default: `16`

### `config.lr_PolynomialLR_power`

:	Power factor for `PolynomialLR`

	Default: `0.999`

## Aesara config

PyCMTensor uses the `aesara` library, which has its own set of configurations. We use the following by default:

`aesara.config.on_unused_input = "ignore"`

`aesara.config.mode = "Mode"`

`aesara.config.allow_gc = False`

Refer to https://aesara.readthedocs.io/en/latest/config.html for other options. 
