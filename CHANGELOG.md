## v0.6.1 (2022-03-08)

### Fix

- re-enable generate_config_file()
- **results**: fix typo in gnorm() function call
- add critial to list of logger functions
- **functions**: compute sum of full ll then divide by shape in neg ll
- **functions**: makes avail in logit() an optional parameter
- **functions**: moved gradient norm function to function.py module
- remove duplicate function

### Refactor

- **database**: rename tensors() to get_tensors()

## v0.6.0 (2022-03-06)

### Feat

- **scheduler.py**: add new scheduler (CyclicLR) for adaptive LR
- **trackers**: add tracking to monitor training loss/score
- **logging**: add logging to __init__
- **params**: add routine to remove unused params from computational graph

### Fix

- **config**: loadout config from train() to configparser
- **utils**: fix TypeError check
- **pycmtensor**: fix various training config
- **logger**: add logging functionality pycmtensor.logger
- **results**: skip precompute routine if H and BHHH already exist
- **expressions**: change dunder calls to aesara.tensor methods
- **models**: compact unused Betas message
- **Beta**: add borrow=True to sharedVariables
- **configparser**: renamed from config to prevent doc conflicts

### Perf

- set fixed learning rate
- **statistics**: precompute H and BHHH to results

## v0.5.0 (2022-03-02)

### Feat

- **config**: add PyCMTensorConfig class to store config settings
- **expressions**: add magic methods lt le gt le ne eq
- **config.py**: enable pre-writing of .aesararc config file on module load
- **models**: add method prob() to MNLogit to output prob slices
- **time_format**: enable logging of build and estimation time

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

### Refactor

- **PyCMTensor**: refactoring models from pycmtensor.py
- **Database**: refactor(Database): refactoring database.py from pycmtensor.py

## v0.4.0 (2022-02-26)

### Perf

- **shared_data**: improve iteration speed by implementing shared() on input data

### Fix

- show excluded data in model est. output
- **results**: standardized naming conventions in modules db->database
- **tqdm**: add arg in train() to enable notebook progressbar

### Feat

- **results**: add Predict class to output probs or discrete choices
- **optimizers**: add AdaGram algorithm

### Refactor

- **optimizers**: refactor base Optimizer class

## v0.3.0 (2022-02-23)

### Feat

- **Database**: add __getattr__ build-in type to Database
- **pycmtensor.py**: add model.output_choices to generate choices

### Refactor

- moved Beta Weights to expressions.py

### Fix

- **swissmetro_test.ipynb**: update swissmetro example
