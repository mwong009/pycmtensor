## Unreleased

### Fix

- make arguments in `MNL` as optional keyword arguments
- moved learning rate variable to `PyCMTensorModel` class

### Refactor

- update `__all__` package variables
- added `train_data` and `valid_data` property to `Data` class

## v1.3.1 (2022-11-17)

### Fix

- fix utility dimensions for asc only cases

## v1.3.0 (2022-11-10)

### Feat

- **optimizers**: added ``Nadam`` optimizer
- **layers.py**: added ``DenseLayer`` ``BatchNormLayer`` ``ResidualLayer``

### Fix

- renamed depreceated instances of ``aesara`` modules
- **data.py**: defaults ``batch_size`` argument to 0 if batch_size is ``None``
- added argument type hints in function.py

### Refactor

- **data**: added import dataset cleaning step as arguments in `Data()`
- moved ResidualLayer to ``pycmtensor.models.layers``
- updated timing to perf_counter
- **pycmtensor**: refactoring model_loglikelihood

## v1.2.1 (2022-10-25)

### Feat

- added ``pycmtensor.about()`` to output package metadata
- added ``pycmtensor.about()`` to output package metadata
- added EMA function ``functions.exp_mov_average()``
- added EMA function ``functions.exp_mov_average()``

### Fix

- updated syntax for ``expressions.py`` class objects
- updated syntax for ``expressions.py`` class objects
- added ``init_type`` property to ``Weights`` class
- added ``init_type`` property to ``Weights`` class
- moved model aesara compile functions from ``models.MNL`` to ``pycmtensor.PyCMTensorModel``
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

### Fix

- **config**: added optimizing speedups to config

### Refactor

- **models**: refactored build_functions() into models.py

## v1.0.6 (2022-08-12)

### Fix

- **config**: set default `cyclic_lr_mode` and `cyclic_lr_step_size` to `None`
- **pre-commit-config**: update black to `22.6.0` in pre-commit check

### Refactor

- **database**: refactor set_choice(choiceVar)

## v1.0.5 (2022-07-27)

### Fix

- **tests**: removed depreciated tests

## v1.0.4 (2022-07-27)

### Fix

- **routine**: remove depreciated tqdm module
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
