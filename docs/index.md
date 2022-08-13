# About PyCMTensor

PyCMTensor is a discrete choice model development platform which is designed with the use of deep learning in mind, enabling users to write more complex models using neural networks.
PyCMTensor is build on [Aesara](https://github.com/aesara-devs/aesara), a tensor library, and uses many features commonly found in deep learning packages such as Tensorflow and Keras.
`Aesara` was chosen as the back end mathematical library because of its hackable, open-source nature.
Users of [Biogeme](https://biogeme.epfl.ch) will be familiar with the syntax of PyCMTensor.

This package allows one to incorporate neural networks into discrete choice models that boosts accuracy of model estimates which still being able to produce all the same statistical analysis found in traditional choice modelling software.

# Download

PyCMTensor is available on PyPi https://pypi.org/project/pycmtensor/. It can be install via

```console
$ pip install -U pycmtensor
```

The latest development version is available via [Github](https://github.com/mwon009/pycmtensor). It can be install via 

```console
$ pip install git+https://github.com/mwong009/pycmtensor.git
```

For more information about installing, see [Installation](installation).

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
