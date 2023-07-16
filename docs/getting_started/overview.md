# Overview

---

Follow the steps below and learn how to use PyCMTensor to estimate a discrete choice model. In this tutorial, we will use the London Passenger Mode Choice (LPMC) dataset ([pdf](https://transp-or.epfl.ch/documents/technicalReports/CS_LPMC.pdf)). Download the dataset [here](http://transp-or.epfl.ch/data/lpmc.dat) and place it in the working directory.

Jump to [Putting it all together](overview.md#putting-it-all-together) for the final Python script.

## Importing data from csv

Import the PyCMTensor package and read the data using `pandas`:

```python
import pycmtensor
import pandas as pd

lpmc = pd.read_csv("lpmc.dat", sep='\t')  # read the .dat file and use <TAB> separator
lpmc = lpmc[lpmc["travel_year"]==2015]  # select only the 2015 data to use
```

#### Create a dataset object

From the `pycmtensor` package, import the `Dataset` object, which stores and manages the tensors and arrays of the data variables. Denote the column name with the choice variable in the argument `choice=`:

```python
from pycmtensor.dataset import Dataset
ds = Dataset(df=lpmc, choice="travel_mode")
```

The `Dataset` object takes the following arguments:

- `df`: The `pandas.DataFrame` object
- `choice`: The name of the choice variable found in the heading of the dataframe

!!! note
    If the range of alternatives in the choice column does not start with `0`, e.g. `[1, 2, 3, 4]` instead of `[0, 1, 2, 3]`, the Dataset will automatically convert the alternatives to start with `0`.


#### Split the dataset

Next, split the dataset into training and validation datasets, `frac=` argument is the percentage of the data that is assigned to the training dataset. The rest of the data is assigned to the validation dataset. 

```python
ds.split(frac=0.8)  # splits 80% of the data into the training dataset
                    # and the other 20% into the validation dataset
```

You should get an output showing the number of training and validation samples in the dataset.

Output:

```bash
[INFO] n_train_samples:3986 n_valid_samples:997
```

!!! note
    Splitting the dataset is optional. If `frac=` is not given as an argument, both training and validation dataset will use the same samples.


## Defining taste parameters

Define the taste parameters using the `Beta` object from the `pycmtensor.expressions` module:

```python
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
- `lb` and `ub`: lower and upper bound of the parameter. Defaults to `None`
- `status`: `1` if the parameter should *not* be estimated. Defaults to `0`.

!!! note
    If a `Beta` variable is not used in the model, a warning will be shown in stdout. E.g.
    
    ```bash
    [WARNING] b_purpose not in any utility functions
    ```

!!! info
    `pycmtensor.expressions.Beta` follows the same syntax as in [Biogeme]() `biogeme.expressions.Beta` for familiarity sake. However, `pycmtensor.expressions.Beta` uses `aesara.tensor` variables to define the mathematical ops. Currently they are not interchangable.


## Specifying utility equations

```python
U_walk  = asc_walk + b_time * ds["dur_walking"]
U_cycle = asc_cycle + b_time  * ds["dur_cycling"]
U_pt    = asc_pt + b_time * (ds["dur_pt_rail"] + ds["dur_pt_bus"] + \
          ds["dur_pt_int"]) + b_cost * ds["cost_transit"]
U_drive = asc_drive + b_time * ds["dur_driving"] + b_licence * ds["driving_license"] + \
          b_cost * (ds["cost_driving_fuel"] + ds["cost_driving_ccharge"])

# vectorize the utility function
U = [U_walk, U_cycle, U_pt, U_drive]
```

We define data variables as an item from the `Dataset` object. For instance, the variable `"dur_walking"` from the LPMC dataset can be expressed as such: `ds["dur_walking"]`. Furthermore, composite variables or interactions can also be specified using standard mathematical operators, for example, adding `"dur_pt_rail"` and `"dur_pt_bus"` can be expressed as `ds["dur_pt_rail"] + ds["dur_pt_bus"]`.

Finally, we vectorize the utility functions by putting them into a `list()`. The index of the utility in the list corresponds to the (zero-adjusted) indexing of the choice variable.

(Advanced, optional) We can also define the utlity functions as a 2-D `tensorVariable` object instead of a list. 

## Specifying the model

To specify the model, we create a model object from a model in the `pycmtensor.models` module.

```python
mymodel = pycmtensor.models.MNL(ds=ds, params=locals(), utility=U, av=None)
```

The `MNL` object takes the following argument:

- `ds`: The dataset object
- `params`: the list (or dict) of declared parameter objects*
- `utility`: The list of utilities to be estimated
- `av`: The availability conditions as a list with the same index as `utility`. See [here]() for an example on specifying availability conditions. Defaults to `None`
- `**kwargs`: Optional keyword arguments for modifying the model configuration settings. See [configuration](../user_guide/configuration.md) in the user guide for details on possible options

!!! tip
    *: We use `locals()` as a shortcut for collecting and fitering the `Beta` objects from the Python [local environment](https://docs.python.org/3/library/functions.html#locals) for the argument `params=`.

Output:

    :::bash 
    [INFO] inputs in MNL: [driving_license, dur_walking, dur_cycling, dur_pt_rail, 
    dur_pt_bus, dur_pt_int, dur_driving, cost_transit, cost_driving_fuel, 
    cost_driving_ccharge]
    [INFO] Build time = 00:00:09
    



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

The function `train()` estimates the model until convergence specified by the gradient norm between two complete passes of the entire training dataset. In order to limit repeated calculation, we store the $\beta$ of the previous epoch and approximate the gradient step using: $\nabla_\beta = \beta_t - \beta_{t-1}$. The estimation is terminated either when the `max_steps` is reached or when the gradient norm $||\nabla_\beta||_{_2}$ is less than the `convergence_threshold` value (set as `0.0001` in this example).

The `train()` function takes the following required arguments:

- `model`: The model object. `MNL` in the example above
- `ds`: The dataset object
- `**kwargs`: Optional keyword arguments for modifying the model configuration settings. See [configuration](../user_guide/configuration) in the user guide for details on possible options

The other arguments `**kwargs` are optional, and they can be set when calling the `train()` function or during model specification. These optional arguments are the so-called *hyperparameters* of the model that modifies the training procedure.

!!! note
    A `step` is one full pass of the training dataset. An `iteration` is one model update operation, usually it is every mini-batch (when `batch_size != 0`).

!!! tip
    The hyperparameters can also be set with the `pycmtensor.config` module before the training function is called.

    For example, to set the training `batch_size` to `50` and `base_learning_rate` to `0.1`:

    ```python
    pycmtensor.config.batch_size = 50
    pycmtensor.config.base_learning_rate = 0.1

    train (
        model=...
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

The results are stored in the `Results` class object of th `MNL` model. The following are function calls to display the statistical results of the model estimation:

```python
print(mymodel.results.beta_statistics())
print(mymodel.results.model_statistics())
print(mymodel.results.benchmark())
```

`beta_statistics()` show the estimated values of the model coefficients, standard errors, t-test, and p-values, including the robust measures.

The standard errors are calculated using the diagonals of the square root of the variance-covariance matrix (the inverse of the negative Hessian matrix):

$$
 std. error = diag.\Big(\sqrt{-H^{-1}}\Big)
$$

The robust standard errors are calculated using the 'sandwich method', where the variance-covariance matrix is as follows:

$$
covar = (-H^{-1})(\nabla\cdot\nabla^\top)(-H^{-1})
$$

The rest of the results are self-explanatory.

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

## Putting it all together

```python linenums="1"
import pycmtensor
import pandas as pd

from pycmtensor.dataset import Dataset
from pycmtensor.expressions import Beta

# read data
lpmc = pd.read_csv("lpmc.dat", sep='\t')
lpmc = lpmc[lpmc["travel_year"]==2015] 

# load data into dataset
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

# utility equations
U_walk  = asc_walk + b_time * ds["dur_walking"]
U_cycle = asc_cycle + b_time  * ds["dur_cycling"]
U_pt    = asc_pt + b_time * (ds["dur_pt_rail"] + ds["dur_pt_bus"] + \
          ds["dur_pt_int"]) + b_cost * ds["cost_transit"]
U_drive = asc_drive + b_time * ds["dur_driving"] + b_licence * ds["driving_license"] + \
          b_cost * (ds["cost_driving_fuel"] + ds["cost_driving_ccharge"])

# vectorize the utility function
U = [U_walk, U_cycle, U_pt, U_drive]

mymodel = pycmtensor.models.MNL(ds=ds, params=locals(), utility=U, av=None)


from pycmtensor.models import train
from pycmtensor.optimizers import Adam
from pycmtensor.scheduler import ConstantLR

# main training loop
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

# print results
print(mymodel.results.beta_statistics())
print(mymodel.results.model_statistics())
print(mymodel.results.benchmark())
```