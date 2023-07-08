# Overview

---

Follow the steps below and learn how to use PyCMTensor to estimate a discrete choice model. In this tutorial, we will use the London Passenger Mode Choice (LPMC) [dataset](). Download the dataset from [here]() and place it in the working directory.

Jump to [Putting it all together](overview.md#putting-it-all-together) for the final Python script.

## Importing data from csv

Import the PyCMTensor package and read the data using `pandas`:

```python
import pycmtensor
import pandas as pd

lpmc = pd.read_csv("lpmc.dat", sep='\t')
lpmc = lpmc[lpmc["travel_year"]==2015]  # select only the 2015 data to use
```

#### Create a dataset object

From the `pycmtensor` package, import the `Dataset` object, which stores and handles the tensors and arrays of the data variables. Denote the column with the choice variable with the argument `choice=`:

```python
from pycmtensor.dataset import Dataset
ds = Dataset(df=lpmc, choice="travel_mode")
```

The `Dataset` object takes the following arguments:

- `df`: The `pandas.DataFrame` object
- `choice`: The name of the choice variable found in the heading of the dataframe

/// note | Note
If the range of alternatives in the choice column does not start with `0`, e.g. `[1, 2, 3, 4]` instead of `[0, 1, 2, 3]`, the Dataset will automatically convert the alternatives to start with `0`.
///

#### Split the dataset

Then, split the dataset into training and validation datasets, `frac=` is the percentage of the data that is assigned to the training dataset. The rest of the data is assigned to the validation dataset. If `frac=` is not given as an argument, both training and validation dataset uses the same total number of samples.

```python
ds.split(frac=0.8)  # splits 80% of the data into the training dataset and 20% into the validation dataset
```

You should get an output showing the number of training and validation samples in the dataset:

    :::bash
    [INFO] n_train_samples:3986 n_valid_samples:997


## Defining taste parameters

Define the taste parameters using the `Beta` object from the `pycmtensor.expressions` module:

```
from pycmtensor.expressions import Beta

# Beta parameters
asc_walk = Beta("asc_walk", 0.0, None, None, 1)
asc_cycle = Beta("asc_cycle", 0.0, None, None, 0)
asc_pt = Beta("asc_pt", 0.0, None, None, 0)
asc_drive = Beta("asc_drive", 0.0, None, None, 0)
b_cost = Beta("b_cost", 0.0, None, None, 0)
b_time = Beta("b_time", 0.0, None, None, 0)
b_purpose = Beta("b_purpose", 0.0, None, None, 0)
b_licence = Beta("b_licence", 0.0, None, None, 0)
```

The `Beta` object takes the following argument:

- `name`: Name of the taste parameter (required)
- `value`: The initial starting value. Defaults to `0.`
- `lb` and `ub`: lower and upper bound. Defaults to `None`
- `status`: `1` if the parameter should *not* be estimated. Defaults to `0`.

/// note | Note
If a `Beta` variable is not used in the model, a warning will be shown in stdout. E.g.
    
    :::bash
    [WARNING] b_purpose not in any utility functions
///

## Specifying utility equations

```python
U_walk = asc_walk + b_time * ds["dur_walking"]
U_cycle = asc_cycle + b_time  * ds["dur_cycling"]
U_pt = asc_pt + b_time * (ds["dur_pt_rail"] + ds["dur_pt_bus"] + ds["dur_pt_int"]) \
       + b_cost * ds["cost_transit"]
U_drive = asc_drive + b_time * ds["dur_driving"] + b_licence * ds["driving_license"] \
          + b_cost * (ds["cost_driving_fuel"] + ds["cost_driving_ccharge"])

# vectorize the utility function
U = [U_walk, U_cycle, U_pt, U_drive]
```

## Specifying the model

```python
mymodel = pycmtensor.models.MNL(ds, locals(), U)
```

Output:

    :::bash
    [WARNING] b_purpose not in any utility functions
    [INFO] inputs in MNL: [driving_license, dur_walking, dur_cycling, dur_pt_rail, dur_pt_bus, dur_pt_int, dur_driving, cost_transit, cost_driving_fuel, cost_driving_ccharge]
    [INFO] Build time = 00:00:09

The `MNL` object takes the following argument:

- `ds`
- `params`
- `utility`
- `av`
- `**kwargs`: Optional keyword arguments for modifying the model configuration settings. See [configuration](../user_guide/configuration.md) for details.

We use `locals()` as a shortcut for collecting the `Beta` objects for the argument `params=`.

## Estimating the model

```python
from pycmtensor.models import train
from pycmtensor.optimizers import Adam
from pycmtensor.scheduler import ConstantLR

train(
    model=mymodel, 
    ds=ds, 
    optimizer=Adam,  # optional
    batch_size=0,  # optional 
    base_learning_rate=1.,  # optional 
    convergence_threshold=0.0001,  # optional 
    max_steps=200,  # optional
    lr_scheduler=ConstantLR,  # optional
)
```

Output:

    :::bash
    [INFO] Start (n=3986, Step=0, LL=-5525.77, Error=80.34%)
    [INFO] Train (Step=0, LL=-9008.61, Error=80.34%, gnorm=2.44949e+00, 0/2000)
    [INFO] Train (Step=16, LL=-3798.26, Error=39.12%, gnorm=4.46640e-01, 16/2000)
    [INFO] Train (Step=54, LL=-3487.97, Error=35.21%, gnorm=6.80979e-02, 54/2000)
    [INFO] Train (Step=87, LL=-3471.01, Error=35.01%, gnorm=9.93509e-03, 87/2000)
    [INFO] Train (Step=130, LL=-3470.29, Error=34.70%, gnorm=1.92617e-03, 130/2000)
    [INFO] Train (Step=168, LL=-3470.28, Error=34.70%, gnorm=3.22536e-04, 168/2000)
    [INFO] Train (Step=189, LL=-3470.28, Error=34.70%, gnorm=8.74120e-05, 189/2000)
    [INFO] Model converged (t=0.492)
    [INFO] Best results obtained at Step 185: LL=-3470.28, Error=34.70%, gnorm=2.16078e-04


## Printing statistical test results

```python
print(mymodel.results.beta_statistics())
print(mymodel.results.model_statistics())
print(mymodel.results.benchmark())
```

Ouput:

    :::bash
                  value   std err     t-test p-value rob. std err rob. t-test rob. p-value
    asc_cycle -3.853007  0.117872 -32.688157     0.0     0.120295  -32.029671          0.0
    asc_drive -2.060414  0.099048 -20.802183     0.0     0.102918  -20.019995          0.0
    asc_pt    -1.305677  0.076151 -17.145988     0.0     0.079729  -16.376401          0.0
    asc_walk        0.0         -          -       -            -           -            -
    b_cost    -0.135635  0.012788 -10.606487     0.0      0.01269  -10.688684          0.0
    b_licence  1.420747  0.079905  17.780484     0.0     0.084526   16.808497          0.0
    b_time    -4.947477  0.183329 -26.986865     0.0     0.192431  -25.710378          0.0
    
                                             value
    Number of training samples used         3986.0
    Number of validation samples used        997.0
    Null. log likelihood              -5525.769323
    Final log likelihood              -3470.282749
    Accuracy                                65.30%
    Likelihood ratio test              4110.973149
    Rho square                            0.371982
    Rho square bar                        0.370715
    Akaike Information Criterion       6954.565498
    Bayesian Information Criterion     6998.599302
    Final gradient norm                2.16078e-04
    
                                value
    Seed                        42069
    Model build time         00:00:09
    Model train time         00:00:00
    iterations per sec  384.15 iter/s

## Prediction and validation

# Putting it all together

```python
import pycmtensor
import pandas as pd

from pycmtensor.dataset import Dataset
from pycmtensor.expressions import Beta

lpmc = pd.read_csv("lpmc.dat", sep='\t')
lpmc = lpmc[lpmc["travel_year"]==2015]  # select only the 2015 data to use

ds = Dataset(df=lpmc, choice="travel_mode")
ds.split(frac=0.8) 

# Beta parameters
asc_walk = Beta("asc_walk", 0.0, None, None, 1)
asc_cycle = Beta("asc_cycle", 0.0, None, None, 0)
asc_pt = Beta("asc_pt", 0.0, None, None, 0)
asc_drive = Beta("asc_drive", 0.0, None, None, 0)
b_cost = Beta("b_cost", 0.0, None, None, 0)
b_time = Beta("b_time", 0.0, None, None, 0)
b_licence = Beta("b_licence", 0.0, None, None, 0)

U_walk = asc_walk + b_time * ds["dur_walking"]
U_cycle = asc_cycle + b_time  * ds["dur_cycling"]
U_pt = asc_pt + b_time * (ds["dur_pt_rail"] + ds["dur_pt_bus"] + ds["dur_pt_int"]) \
       + b_cost * ds["cost_transit"]
U_drive = asc_drive + b_time * ds["dur_driving"] + b_licence * ds["driving_license"] \
          + b_cost * (ds["cost_driving_fuel"] + ds["cost_driving_ccharge"])

# vectorize the utility function
U = [U_walk, U_cycle, U_pt, U_drive]

mymodel = pycmtensor.models.MNL(ds, locals(), U)


from pycmtensor.models import train
from pycmtensor.optimizers import Adam
from pycmtensor.scheduler import ConstantLR

train(
    model=mymodel, 
    ds=ds, 
    optimizer=Adam,  # optional
    batch_size=0,  # optional 
    base_learning_rate=1.,  # optional 
    convergence_threshold=0.0001,  # optional 
    max_steps=200,  # optional
    lr_scheduler=ConstantLR,  # optional
)

print(mymodel.results.beta_statistics())
print(mymodel.results.model_statistics())
print(mymodel.results.benchmark())
```