## v0.5.0b0 (2022-03-01)

### Fix

- **result**: print more verbose results and options

### Feat

- **models**: add method prob() to MNLogit to output prob slices
- **time_format**: enable logging of build and estimation time

## v0.4.1 (2022-02-27)

### Fix

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

## v0.2.1 (2022-02-20)
