# About PyCMTensor

PyCMTensor is a discrete choice modelling development tool on deep learning libraries, enabling development of complex models using deep neural networks.
PyCMTensor is build on [Aesara](https://github.com/aesara-devs/aesara), a tensor library which uses features commonly found in deep learning packages such as Tensorflow and Keras.
``Aesara`` was chosen as the back end mathematical library because of its hackable, open-source nature.
Users of [Biogeme](https://biogeme.epfl.ch) would be familiar with the syntax of PyCMTensor.

# Download

PyCMTensor is available on PyPi https://pypi.org/project/pycmtensor/. It can be install through

```console
$ pip install -U pycmtensor
```

The latest development version is available via [Github](https://github.com/mwon009/pycmtensor). It can be install via 

```console
$ pip install git+https://github.com/mwong009/pycmtensor.git
```

For more information about installing, see [Installation](installation).

# Overview

PyCMTensor is written similar to [Biogeme](https://biogeme.epfl.ch). 

To import a dataset, for example the ``swissmetro.dat`` dataset:

```python
import pycmtensor as cmt
swissmetro = pd.read_csv("swissmetro.dat", sep="\t")
db = cmt.Data(df=swissmetro, choice="CHOICE")
db.split_db(split_frac=0.8)  # split dataset into train and valid datasets
```

Define the choice model coefficients and utility equations:

```python
b_cost = Beta("b_cost", 0.0, None, None, 0)
b_time = Beta("b_time", 0.0, None, None, 0)
asc_train = Beta("asc_train", 0.0, None, None, 0)
asc_car = Beta("asc_car", 0.0, None, None, 0)
asc_sm = Beta("asc_sm", 0.0, None, None, 1)

U_1 = b_cost * db["TRAIN_CO"] + b_time * db["TRAIN_TT"] + asc_train
U_2 = b_cost * db["SM_CO"] + b_time * db["SM_TT"] + asc_sm
U_3 = b_cost * db["CAR_CO"] + b_time * db["CAR_TT"] + asc_car

# specify the utility function and the availability conditions
U = [U_1, U_2, U_3]  # utility
AV = [db["TRAIN_AV"], db["SM_AV"], db["CAR_AV"]]  # availability

mymodel = MNL(U, AV, locals(), db, name="MNL")
```

Train or execute model estimation

```python
mymodel.train(db)
```

**Generate results and statistics**
```python
print(mymodel.results.model_statistics())
```
```
                                          value
Number of training samples used          8575.0
Number of validation samples used        2143.0
Init. log likelihood               -8869.978759
Final log likelihood               -7532.283616
Accuracy                                 60.15%
Likelihood ratio test               2675.390285
Rho square                             0.150812
Rho square bar                         0.150248
Akaike Information Criterion       15074.567232
Bayesian Information Criterion     15109.850263
Final gradient norm                    0.005921
```

**Beta statistics**
```python
print(mymodel.results.beta_statistics())
```
```
              value   std err     t-test   p-value rob. std err rob. t-test rob. p-value
asc_car   -0.681302   0.04447 -15.320491       0.0     0.136012   -5.009141     0.000001
asc_sm          0.0         -          -         -            -           -            -
asc_train -1.615423  0.047496 -34.011958       0.0     0.147135  -10.979158          0.0
b_cost     0.030627  0.019547   1.566831  0.117154     0.010567    2.898396     0.003751
b_time    -0.303005  0.048987   -6.18539       0.0     0.160293    -1.89032     0.058715
```

**Correlation matrix**
```python
print(mymodel.results.model_correlation_matrix())
```
```
             b_cost    b_time  asc_train   asc_car
b_cost     1.000000  0.209228   0.226905 -0.034368
b_time     0.209228  1.000000   0.730218  0.790017
asc_train  0.226905  0.730218   1.000000  0.660197
asc_car   -0.034368  0.790017   0.660197  1.000000
```

**Choice probability predictions**
```python
print(mymodel.predict(db, return_choices=False))
```
```
[[0.12677006 0.5450971  0.32813285]
 [0.12620901 0.54609865 0.32769234]
 [0.12715643 0.54322562 0.32961795]
 ...
 [0.13161936 0.5223758  0.34600483]
 [0.1307124  0.51879211 0.35049549]
 [0.13236051 0.52069128 0.34694821]]
 ```

# Documentation

- [Introduction](introduction)
- [Installation](installation)
- [Usage](usage)
- [Development Guide](development)
- [Changelog](changelog)
- [API Reference](autoapi/index)

```{toctree} 
:caption: User guide
:maxdepth: 3
:hidden:

introduction
installation
usage
development
authors
changelog
autoapi/index
```

---

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
