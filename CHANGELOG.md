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
