## v1.8.1 (2023-11-28)

### Fix

- **basic**: add lr_scheduler to `compute()` function

### Refactor

- clean up and enhance code with codiumAI

## v1.8.0 (2023-11-24)

### Feat

- **utils**: add new function to display numbers as human readable format
- **results**: add new function to plot training statistics

### Fix

- **scheduler**: fix missing arguments
- **basic**: renamed variable to avoid confusion

## v1.7.1 (2023-09-28)

### Fix

- **scheduler**: fixed calculation erro in cyclic lr
- condition for stdout

## v1.7.0 (2023-09-22)

### Feat

- new option for selection of model acceptance pattern
- calculating distributed Beta now uses percentile instead of mean
- **models**: add new func `compute` to run the model on a manual set of params
- **regularizers**: new method `include_regularization_terms()` in BaseModel for adding regularizers

### Fix

- **dependencies**: prevent breakage of numpy 1.26
- import compute from pycmtensor.models
- fix l2 regularizer formula
- set init_types as a module level list and add tests
- add minmax clipping on neural net layer outputs
- temporary function for negative relu
- expression fix for random draw variables

### Refactor

- relax the condition for stdout during training
- **config**: refactor config.add into .add() and .update() methods

## v1.6.4 (2023-08-14)

### Refactor

- changed basic model functions into staticmethod
- **expressions.py**: make ExpressionParser.parse a staticmethod
- remove depreciated modules

## v1.6.3 (2023-08-11)

### Fix

- **expressions.py**: use predefined draw types for random draws
- **basic.py**: include choice label(s) as a dictionary for `predict()`
- moved elasticities from statistics to model

## v1.6.2 (2023-08-10)

## v1.6.1 (2023-08-10)

### Fix

- **scheduler.py**: add learning rate lower bounds for decaying functions
- **basic.py**: add placeholder arguments `*args`

### Refactor

- **basic.py**: improve efficency of hessian matrix calculation over the sum of log likelihood over observation
- **basic.py**: refactoring common model functions into BaseModel
- syntax and naming changes

## v1.6.0 (2023-08-07)

### Feat

- **optimizers.py**: include new optimizer RProp
- **functions.py**: speed up computation and compilation by using static indexing in `log_likelihood` function
- **functions.py**: add `relu()` function (source taken from Theano 0.7.1)
- **basic.py**: new function include_params_for_convergence

### Fix

- **__init__.py**: fix init circular imports
- **statistics.py**: update t_test and varcovar matrix calculations for vector parameters
- **layers.py**: various fixes to neural net layers
- **optimizers.py**: fix SQNBFGS algorithm
- **functions.py**: include output as params for 1st and 2nd order derivatives
- **expressions.py**: fix base class inheritence and use `config.seed` for seed value
- **expressions.py**: include `self` in overloaded operators if `other` instance is of similar type as `self`
- **basic.py**: fix incorrect saved params in `train()`

### Refactor

- **results.py**: updates results calculations for beta vectors
- **expressions.py**: add set_value in `Param` base class

## v1.5.0 (2023-07-26)

### Feat

- **dataset.py**: use `as_tensor_variable` to construct tensor vector from dataset[[item1, item2,...]]
- **dataset.py**: added list(`str`) as tensor arguments for `train_dataset()` and `valid_dataset()`

### Fix

- **expressions.py**: clip lower and upper bounds when updating `Betas`
- **expressions.py**: fixed `Weights` and `Bias` mathematical operations
- **config.py**: renamed config.py to defaultconfig.py to avoid name conflicts

### Refactor

- **optimizers.py**: removed unused imports

## v1.4.0 (2023-07-16)

### Feat

- **optimizers.py**: added new BFGS algorithm
- **dataset.py**: moved dataset initialization to dataset.py
- **pycmtensor.py**: Implemented early stopping on coefficient convergence in training loop
- **functions.py**: logit method now takes uneven dimensioned-utilities
- **expression.py**: Added RandomDraws expression for sampling in mixed logit
- get_train_data optional argument numpy_out to return numpy arrays rather than pandas arrays
- BHHH algorithm for calculating var-covar matrix applies to each data row

### Fix

- **results.py**: fixed instance when params are not Betas
- **expressions.py**: include RandomDraws as an option in expression evaluations
- Update tests workflow file conda packages
- **pycmtensor.py**: Added missing configuration for stepLR `drop_every`
- **tests.yml**: Update tests workflow file conda packages
- **optimizers.py**: Fixed name typo in `__all__`
- **results.py**: Corrected calculation of hessian and bhhh matrices
- **scheduler.py**: Moved class function calls to parent class
- **statistics.py**: Fixed rob varcovar calculation error
- **MNL.py**: Moved aesara function to parent class
- **data.py**: Streamlined class function calls and removed unnecessary code
- removed package import clashes with config.py
- removed gnorm calculation
- update hessian matrix and bhhh algorithm functions

### Refactor

- **pycmtensor.py**: temporarily removed pycmtensor.py
- **MNL.py**: replaced function constructors from pycmtensor.py as a function call inside the model Class object
- **basic.py**: moved model functionality to from pycmtensor.py to models/basic.py
- **utils.py**: Removed unused code

## v1.3.2 (2023-06-23)

### Fix

- make arguments in `MNL` as optional keyword arguments
- moved learning rate variable to `PyCMTensorModel` class

### Refactor

- make model variables as property
- update `__all__` package variables
- added `train_data` and `valid_data` property to `Data` class

## v1.3.1 (2022-11-17)

### Fix

- fix utility dimensions for asc only cases

## v1.3.0 (2022-11-10)

### Feat

- **optimizers**: added ``Nadam`` optimizer
- **layers.py**: added ``DenseLayer`` ``BatchNormLayer`` ``ResidualLayer``
- added ``pycmtensor.about()`` to output package metadata
- added EMA function ``functions.exp_mov_average()``

### Fix

- renamed depreceated instances of ``aesara`` modules
- **data.py**: defaults ``batch_size`` argument to 0 if batch_size is ``None``
- updated syntax for ``expressions.py`` class objects
- added ``init_type`` property to ``Weights`` class
- moved model aesara compile functions from ``models.MNL`` to ``pycmtensor.PyCMTensorModel``
- added argument type hints in function.py

### Refactor

- **data**: added import dataset cleaning step as arguments in `Data()`
- moved ResidualLayer to ``pycmtensor.models.layers``
- updated timing to perf_counter
- **pycmtensor**: refactoring model_loglikelihood

## v1.2.1 (2022-10-25)

### Feat

- added ``pycmtensor.about()`` to output package metadata
- added EMA function ``functions.exp_mov_average()``

### Fix

- updated syntax for ``expressions.py`` class objects
- added ``init_type`` property to ``Weights`` class
- moved model aesara compile functions from ``models.MNL`` to ``pycmtensor.PyCMTensorModel``

## v1.2.0 (2022-10-14)

### Feat

- **expressions**: added Weights class object (#59)
- **functions**: added rmse and mae objective functions (#58)
- batch shuffle for training
- **function**: added KL divergence loss function (#50)

### Fix

- added expand_dims into logit function
- replace class function Beta.Beta with Beta.beta
- removed flatten() from logit function

## v1.1.0 (2022-09-23)

### Feat

- **scheduler**: added learning rate scheduling to train()
- **code**: overhaul and cleanup

### Fix

- **environment**: update project deps and pre-commit routine
- **config**: remove unnecessary cxx flags from macos builds

### Perf

- **config**: misc optimization changes

## v1.0.7 (2022-08-12)

## v1.0.6 (2022-08-12)

### Fix

- **config**: added optimizing speedups to config
- **config**: set default `cyclic_lr_mode` and `cyclic_lr_step_size` to `None`
- **pre-commit-config**: update black to `22.6.0` in pre-commit check

### Refactor

- **models**: refactored build_functions() into models.py
- **database**: refactor set_choice(choiceVar)

## v1.0.5 (2022-07-27)

### Fix

- **tests**: removed depreciated tests
- **routine**: remove depreciated tqdm module

## v1.0.4 (2022-07-27)

### Fix

- **pycmtensor.py**: update training method
- **config.py**: new config option verbosity: "high", "low"
- **pycmtensor.py**: remove warnings for max_iter<patience

## v1.0.3 (2022-05-12)

## v1.0.2 (2022-05-12)

## v1.0.1 (2022-05-12)

### Fix

- **scheduler**: fix missing args in input parameters
- **scheduler**: fix constantLR missing input paramerer

## v1.0.0 (2022-05-10)

### Feat

- **python**: update to python 3.10

### Fix

- **tests**: update tests files to reflect changes in biogeme removal

## v0.8.0 (2022-05-10)

### Feat

- **deps**: remove Biogeme dependencies

## v0.7.1 (2022-05-10)

### Fix

- **expressions**: remove Biogeme dependencies
- **database**: remove dependencies of Biogeme
- **debug**: remove debug handler after each run to prevent duplication
- **models**: add function to return layer output -> get_layer_outputs()
- **debug**: disables tqdm if debug mode is on and activates debug_log

### Refactor

- move elasticites from models to statistics for consistency

## v0.7.0 (2022-03-17)

### Feat

- **models**: add functionality to compute elasticities of choice vs attribute in models.py

### Fix

- **results**: remove unnessary `show_weights` option in Results
- set default max_epoch on training run to adaptive rule
- print valid config options when invalid options are given as args to train()
- **scheduler**: modified cyclic_lr config loading sequence to fix unboundError
- **train**: turn saving model off for now
- **config**: generate os dependent ld_flags

### Refactor

- **utils**: refactored save_to_pickle and disables it

### Perf

- **IterationTracker**: use numpy array to store iteration data

## v0.6.5 (2022-03-14)

### Feat

- **models**: Implement the ResLogit layer

### Fix

- **config**: set default learning schedule to ConstantLR
- **config**: set default seed to a random number on init

## v0.6.4 (2022-03-13)

### Feat

- **scheduler.py**: add new scheduler (CyclicLR) for adaptive LR

### Fix

- **project**: fix project metadata and ci
- **config**: loadout config from train() to configparser
- **utils**: fix TypeError check

## v0.5.0 (2022-03-02)

### Feat

- **config**: add PyCMTensorConfig class to store config settings
- **expressions**: add magic methods lt le gt le ne eq
- **config.py**: enable pre-writing of .aesararc config file on module load
- **models**: add method prob() to MNLogit to output prob slices
- **time_format**: enable logging of build and estimation time
- **results**: add Predict class to output probs or discrete choices
- **optimizers**: add AdaGram algorithm
- **Database**: add __getattr__ build-in type to Database
- **pycmtensor.py**: add model.output_choices to generate choices

### Fix

- **statistics**: add small value to stderror calculation to address sqrt(0)
- **dependencies**: move ipywidgets and pydot to dependencies
- renamed .rst to .md fix FileNotFoundError
- **result**: print more verbose results and options
- **Database**: add name to shared_data
- **train**: model instance now load initiated model class (not input Class as argument)
- **Database**: set choiceVar to mandatory argument
- **PyCMTensor**: rename append_to_params to add_params for consistency
- **PyCMTensor**: new method to add regularizers to cost function
- **Expressions**: invokes different operator for Beta Beta maths
- show excluded data in model est. output
- **results**: standardized naming conventions in modules db->database
- **tqdm**: add arg in train() to enable notebook progressbar
- **swissmetro_test.ipynb**: update swissmetro example

### Refactor

- **PyCMTensor**: refactoring models from pycmtensor.py
- **Database**: refactor(Database): refactoring database.py from pycmtensor.py
- **optimizers**: refactor base Optimizer class
- moved Beta Weights to expressions.py

### Perf

- **shared_data**: improve iteration speed by implementing shared() on input data
