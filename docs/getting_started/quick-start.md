# Quick start guide

---

Follow the steps below and learn how to use PyCMTensor to estimate a discrete choice model. In this tutorial, we will use the London Passenger Mode Choice (LPMC) dataset ([pdf](https://transp-or.epfl.ch/documents/technicalReports/CS_LPMC.pdf)). Download the dataset [here](http://transp-or.epfl.ch/data/lpmc.dat) and place it in the working directory.

Jump to [Putting it all together](quick-start.md#putting-it-all-together) for the final Python script.

## Importing data from csv

Import the PyCMTensor package and read the data using `pandas`:

```python
import pycmtensor
import pandas as pd

lpmc = pd.read_csv("lpmc.dat", sep='\t') 
```

Optional: Using Pandas, you can make modifications and selections to the dataset using indexing operators `[]` and `.` for example, selecting data from `"travel_year"==2015`:

```python
lpmc = lpmc[lpmc["travel_year"]==2015]
```

#### Create a dataset object

From the `pycmtensor` package, import the `Dataset` object, which stores and manages the tensors and arrays of the data variables. Denote the column name with the choice variable in the argument `choice=`:

```python
from pycmtensor.dataset import Dataset
ds = Dataset(df=lpmc, choice="travel_mode")
```

The `Dataset` object takes the following arguments:

- `df`: The `pandas.DataFrame` object
- `choice`: `str` The name of the choice variable found in the heading of the dataframe

Optional arguments are:

- `shuffle`: `[True|False]`. Whether the dataset should be shuffled or not

!!! note
    The range of alternatives should begin with `0`, if not, an error might be raised during model training.

#### (Optional) Scaling variables

Before proceeding with model training, it might be beneficial to scale your variables, especially if they have different ranges. This can help the model converge faster and result in a better performance. PyCMTensor provides a built in method for scaling, for example:

```python
# scaling the walking duration variable by 100 `(scaled_x = x/100)`
ds.scale_variable("dur_walking", 100)
```

alternatively, by using Pandas _before_ creating the `Dataset` object:

```python
lpmc["dur_walking"] = lpmc["dur_walking"].div(100)
ds = Dataset(df=lpmc, choice="travel_mode")
```

Note: Scaling is optional and might not always result in better performance. It's recommended to experiment with and without scaling to see what works best for your specific use case.


#### Split the dataset
\
Next, we'll divide the dataset into training and validation subsets. The `frac=` parameter determines the proportion of data allocated to the training set, with the remainder going to the validation set.


```python
ds.split(frac=0.8)  # Allocates 80% of the data to the training set
                    # The remaining 20% is used for validation
```

Alternatively, you can specify the number of samples for the training set using the count parameter:

```python
ds.split(count=3986)  # Sets aside 3986 samples for training
```

Upon execution, you should see an output indicating the number of samples in the training and validation sets.

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
asc_walk = Beta("asc_walk", 0.0, None, None, 1) # do not estimate asc_walk
asc_cycle = Beta("asc_cycle", 0.0, None, None, 0)
asc_pt = Beta("asc_pt", 0.0, None, None, 0)
asc_drive = Beta("asc_drive", 0.0, None, None, 0)
b_cost = Beta("b_cost", 0.0, None, None, 0)
b_time = Beta("b_time", 0.0, None, None, 0)
b_purpose = Beta("b_purpose", 0.0, None, None, 0)
b_licence = Beta("b_licence", 0.0, None, None, 0)
```

The `Beta(name=, value=, lb=, ub=, status=)` object takes the following argument:

- `name`: Name of the taste parameter (required)
- `value`: The initial starting value. Defaults to `0.`
- `lb` and `ub`: lower and upper bound of the parameter. Defaults to `None`
- `status`: `[0|1]` set to `1` if the parameter should *not* be estimated. Defaults to `0`.

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

Data variables are defined as items from the `Dataset` object. For example, the `"dur_walking"` variable from the LPMC dataset is represented as `ds["dur_walking"]`. Additionally, you can specify composite variables or interactions using standard mathematical operators. An example of this would be the addition of `"dur_pt_rail"` and `"dur_pt_bus"`, represented as `ds["dur_pt_rail"] + ds["dur_pt_bus"]`.

Utility functions are then vectorized by placing them into a `list()`. The utility's position in the list corresponds to the choice variable's (zero-adjusted) index.

For advanced users, there's an optional method where utility functions can be defined as a 2-D tensorVariable object instead of a list [TODO].

## Specifying the model

To define the model, we instantiate a model object using a model from the `pycmtensor.models` module.

```python
from pycmtensor.models import MNL
mymodel = MNL(ds=ds, variables=locals(), utility=U, av=None)
```

The `MNL` object takes the following argument:

- `ds`: Represents the dataset object
- `variables`: This is a list (or dictionary) of declared parameter objects
- `utility`: This is a list of utilities that need to be estimated
- `av`: These are the availability conditions, listed in the same order as `utility`. For an example on how to specify availability conditions, refer to [here](). If not specified, defaults to `None`.
- `**kwargs`: These are optional keyword arguments used to modify the model configuration settings. For details on possible options, refer to the [configuration](../user_guide/configuration.md) section in the user guide.

!!! tip
    *: We use `locals()` as a shortcut to gather and filter the `Beta` objects from the Python [local environment](https://docs.python.org/3/library/functions.html#locals) for the argument `variables=`.

Example output:

```bash 
[INFO] choice: travel_mode
[INFO] inputs in MNL: [driving_license, dur_walking, dur_cycling, dur_pt_rail, 
dur_pt_bus, dur_pt_int, dur_driving, cost_transit, cost_driving_fuel, 
cost_driving_ccharge]
[INFO] Build time = 00:00:01
```

## Estimating the model

Here's a basic outline of the steps involved in estimating a model:

```python
from pycmtensor import train
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

The `train()` function is used to estimate the model until a specified convergence is achieved. This convergence is determined by the gradient norm between two full iterations over the entire training dataset. To avoid redundant calculations, we keep track of the $\beta$ value from the previous iteration and approximate the gradient step using the formula: $\nabla_\beta = \beta_t - \beta_{t-1}$. The estimation process is halted when either the maximum number of steps (max_steps) is reached or when the gradient norm $||\nabla_\beta||_{_2}$ falls below a certain convergence_threshold value (which is 0.0001 in this example).

The `train()` function takes the following required arguments:

- `model`: The model object. `MNL` in the example above
- `ds`: The dataset object
- `**kwargs`: These are optional keyword arguments used to modify the model configuration settings. For details on possible options, refer to the [configuration](../user_guide/configuration.md) section in the user guide.

The other arguments `**kwargs` are optional, and they can be set when calling the `train()` function or during model specification. These optional arguments are the so-called *hyperparameters* of the model that modifies the training procedure.

The additional arguments, denoted as `**kwargs`, are optional and can be specified either when invoking the `train()` function or during the model definition. These optional parameters, known as *hyperparameters*, alter the training process.

!!! note
    A step in `max_step` refers to a complete traversal of the training dataset. An iteration is a single model update operation, typically performed on every mini-batch (when `batch_size` != 0).

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

Example output:

    :::bash
    [INFO] Start (n=3986, Step=0, LL=-5525.77, Error=80.34%)
    [INFO] Train (Step=0, LL=-9008.61, Error=80.34%, gnorm=2.44949e+00, 0/2000)
    [INFO] Train (Step=16, LL=-3798.26, Error=39.12%, gnorm=4.46640e-01, 16/2000)
    [INFO] Train (Step=54, LL=-3487.97, Error=35.21%, gnorm=6.80979e-02, 54/2000)
    ...
    [INFO] Train (Step=189, LL=-3470.28, Error=34.70%, gnorm=8.74120e-05, 189/2000)
    [INFO] Model converged (t=0.492)
    [INFO] Best results obtained at Step 185: LL=-3470.28, Error=34.70%, gnorm=2.16078e-04


## Results analysis

The results are stored in the `Results` class object of th `MNL` model. The following are function calls to display the statistical results of the model estimation:

```python
print(mymodel.results.beta_statistics())
print(mymodel.results.model_statistics())
print(mymodel.results.benchmark())
```

### Beta statistics 

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

`mymodel.results.beta_statistics()`:

    :::bash
               value std err  t-test p-value rob. std err rob. t-test rob. p-value
    asc_cycle -3.956   0.082 -48.336     0.0        0.082     -48.495          0.0
    asc_drive -2.956   0.092 -32.048     0.0        0.094     -31.382          0.0
    asc_pt    -1.379   0.047 -29.597     0.0        0.049     -28.017          0.0
    asc_walk     0.0       -       -       -            -           -            -
    b_cost    -0.141   0.013 -10.945     0.0        0.012     -11.474          0.0
    b_licence  1.464   0.081  18.156     0.0        0.086       17.09          0.0
    b_purpose  0.291   0.032    9.15     0.0        0.032       8.976          0.0
    b_time    -4.971   0.182 -27.333     0.0        0.195     -25.517          0.0

### Model statistics

This section presents the model's performance metrics, which include the log likelihood, training and validation accuracy, rho square, Akaike Information Criterion (AIC), and Bayesian Information Criterion (BIC).

`mymodel.results.model_statistics()`:
    
    :::bash
                                                     value
    Number of training samples used                   3986
    Number of validation samples used                  997
    Number of estimated parameters in the model          8
    Null. log likelihood                          -5525.77
    Final log likelihood                          -3414.87
    Validation Accuracy                             64.69%
    Training Accuracy                               64.40%
    Likelihood ratio test                          4221.80
    Rho square                                       0.382
    Rho square bar                                   0.381
    Akaike Information Criterion                   6845.74
    Bayesian Information Criterion                 6896.07
    Final gradient norm                          9.700e-04
    Maximum epochs reached                              No
    Best result at epoch                               243
    
### Benchmark

This section presents the model performance on the given hardware

`mymodel.results.benchmark()`:

    :::bash
                                                           value
    Platform                                  Windows 10.0.22631
    Processor           Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz
    RAM                                                 15.85 GB
    Python version                                        3.11.7
    PyCMTensor version                                    1.10.0
    Seed                                                     100
    Model build time                                    00:00:01
    Model train time                                    00:00:01
    epochs per sec                                207.84 epoch/s


## Correlation and robust correlation matrix

PyCMTensor provides two built-in methods to compute the correlation and robust correlation matrix `results.model_correlation_matrix()` or `results.model_robust_correlation_matrix()` respectively.
Both the correlation matrix and the robust correlation matrix can provide valuable insights into the relationships between the $\beta$ parameters of the model. 

```python
print(mymodel.results.model_correlation_matrix())
print(mymodel.results.model_robust_correlation_matrix())
```

Output:

    :::bash
               asc_walk  asc_cycle  asc_pt  asc_drive  b_cost  b_time  b_purpose  b_licence
    asc_walk      1.000     -0.280   0.135     -0.553   0.049  -0.644      0.433      0.307
    asc_cycle    -0.280      1.000  -0.352     -0.505   0.128   0.035      0.254      0.132
    asc_pt        0.135     -0.352   1.000     -0.292  -0.186   0.245      0.457      0.265
    asc_drive    -0.553     -0.505  -0.292      1.000  -0.055   0.317     -0.773     -0.476
    b_cost        0.049      0.128  -0.186     -0.055   1.000  -0.018     -0.025     -0.056
    b_time       -0.644      0.035   0.245      0.317  -0.018   1.000     -0.091     -0.143
    b_purpose     0.433      0.254   0.457     -0.773  -0.025  -0.091      1.000      0.026
    b_licence     0.307      0.132   0.265     -0.476  -0.056  -0.143      0.026      1.000

## Prediction and validation=

PyCMTensor produces a probability vector for each entry in the (validation) dataset when predicting choices, followed by the predicted choice and the ground truth data. To display the estimated prediction probabilities, utilize the following function:

```python
prob = mymodel.predict(ds)
print(pd.DataFrame(prob))
```

Output:
```console
                0         1         2         3  pred_travel_mode  true_travel_mode
0    1.823828e-02  0.026140  0.114631  0.840990                 3                 3
1    3.076091e-02  0.015439  0.128448  0.825352                 3                 2
2    6.249183e-01  0.019045  0.264769  0.091267                 0                 2
3    3.877266e-08  0.002577  0.073290  0.924132                 3                 3
4    5.058005e-05  0.025173  0.827738  0.147039                 2                 3
..            ...       ...       ...       ...               ...               ...
992  3.490210e-01  0.013751  0.159579  0.477649                 3                 0
993  7.535218e-03  0.028429  0.107126  0.856910                 3                 3
994  2.813242e-01  0.014489  0.175488  0.528699                 3                 3
995  4.005819e-05  0.005061  0.011871  0.983029                 3                 3
996  1.740788e-02  0.030565  0.183020  0.769007                 3                 3

[997 rows x 6 columns]
```

## Elasticities

Disaggregated elasticities are generated from the model by specifiying the dataset and the reference choice as `wrt_choice`. For instance, to compute the elasticities of the mode for driving (`wrt_choice=3`) over all the input variables:

```python
elas = mymodel.elasticities(ds=ds, wrt_choice=3)
print(pd.DataFrame(elas).round(3))
```

Output:
```console
      cost_driving_ccharge  cost_driving_fuel  cost_transit  driving_license  dur_cycling  \
0                     -0.0             -0.033         0.063            0.228        0.022   
1                     -0.0             -0.074         0.159            0.716        0.022   
2                     -0.0             -0.110         0.189            0.000        0.029   
3                     -0.0             -0.032         0.000            0.793        0.015   
4                     -0.0             -0.089         0.114            0.000        0.031   
...                    ...                ...           ...              ...          ...   
3981                  -0.0             -0.023         0.000            0.000        0.008   
3982                  -0.0             -0.131         0.000            0.798        0.038   
3983                  -0.0             -0.018         0.050            0.565        0.016   
3984                  -0.0             -0.012         0.040            0.162        0.019   
3985                  -0.0             -0.042         0.114            0.000        0.026   

      dur_driving  dur_pt_bus  dur_pt_int  dur_pt_rail  dur_walking  purpose  
0          -0.193       0.287       0.099        0.000        0.003    0.045  
1          -1.033       0.000       0.000        0.312        0.063    0.712  
2          -1.159       0.000       0.000        0.459        0.062    0.710  
3          -0.482       0.000       0.000        0.107        0.297    0.473  
4          -0.561       0.062       0.249        0.560        0.000    0.224  
...           ...         ...         ...          ...          ...      ...  
3981       -0.199       0.081       0.000        0.000        0.556    1.073  
3982       -1.604       0.324       0.221        0.663        0.009    0.476  
3983       -0.263       0.168       0.000        0.000        0.376    0.337  
3984       -0.117       0.039       0.070        0.047        0.033    0.161  
3985       -0.483       0.227       0.000        0.000        0.724    0.247  

[3986 rows x 11 columns]
```

The aggregated elasticities can then be obtained by taking the mean over the rows

```python
import numpy as np 
np.mean(pd.DataFrame(elas), axis=0)
```

Aggregated elasticities (w.r.t driving mode):

```console
cost_driving_ccharge   -0.158813
cost_driving_fuel      -0.063062
cost_transit            0.088155
driving_license         0.407574
dur_cycling             0.042154
dur_driving            -0.794599
dur_pt_bus              0.225808
dur_pt_int              0.079546
dur_pt_rail             0.245628
dur_walking             0.333738
purpose                 0.441163
```

## Putting it all together

```python linenums="1"
import pandas as pd
from pycmtensor.dataset import Dataset
from pycmtensor.expressions import Beta, RandomDraws
from pycmtensor.models import MNL

# read data
lpmc = pd.read_csv("../data/lpmc.dat", sep="\t")
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
s_time = Beta("s_time", 0.5, None, None, 0)
b_purpose = Beta("b_purpose", 0.0, None, None, 0)
b_licence = Beta("b_licence", 0.0, None, None, 0)
b_car_own = Beta("b_car_own", 0.0, None, None, 0)

# utility equations
U_walk = asc_walk + b_time * ds["dur_walking"]
U_cycle = asc_cycle + b_time * ds["dur_cycling"] 
U_pt = asc_pt + b_time * (ds["dur_pt_rail"] + ds["dur_pt_bus"] + ds["dur_pt_int"]) + b_cost * ds["cost_transit"]
U_drive = asc_drive + b_time * ds["dur_driving"] + b_licence * ds["driving_license"] + b_cost * (ds["cost_driving_fuel"] + ds["cost_driving_ccharge"])  + b_purpose * ds["purpose"]

# vectorize the utility function
U = [U_walk, U_cycle, U_pt, U_drive]

mymodel = MNL(ds, locals(), U)

### Training the model ###

from pycmtensor import train
from pycmtensor.optimizers import Adam
from pycmtensor.scheduler import ConstantLR

# main training loop
mymodel.reset_values()
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

# display results
print(mymodel.results.beta_statistics())
print(mymodel.results.model_statistics())
print(mymodel.results.benchmark())

# correlation matrix
print(mymodel.results.model_correlation_matrix())
print(mymodel.results.model_robust_correlation_matrix())

# predictions
prob = mymodel.predict(ds)
print(pd.DataFrame(prob))

# elasticities
elas = mymodel.elasticities(ds=ds, wrt_choice=3)
print(pd.DataFrame(elas).round(3))

import numpy as np 
print(np.mean(pd.DataFrame(elas), axis=0))
```