# test_optimizers.py
import pytest

import pycmtensor.optimizers as opt


def test_sgd(mnl_model):
    sgd = opt.SGD(mnl_model.params)
    assert repr(sgd) == "SGD"

    updates = sgd.update(mnl_model.cost, mnl_model.params)
    assert isinstance(updates, list)
    assert len(updates) == 4


def test_adagrad(mnl_model):
    adagrad = opt.AdaGrad(mnl_model.params)
    assert repr(adagrad) == "AdaGrad"

    updates = adagrad.update(mnl_model.cost, mnl_model.params)
    assert isinstance(updates, list)
    assert len(updates) == 8


def test_momentum(mnl_model):
    nesterov = opt.Momentum(mnl_model.params)
    assert repr(nesterov) == "NAG"
    updates = nesterov.update(mnl_model.cost, mnl_model.params)

    momentum = opt.Momentum(mnl_model.params, nesterov=False)
    assert repr(momentum) == "Momentum"
    updates = momentum.update(mnl_model.cost, mnl_model.params)
    assert isinstance(updates, list)
    assert len(updates) == 8


def test_rmsprop(mnl_model):
    rmsprop = opt.RMSProp(mnl_model.params)
    assert repr(rmsprop) == "RMSProp"
    updates = rmsprop.update(mnl_model.cost, mnl_model.params)
    assert len(updates) == 8


def test_adadelta(mnl_model):
    adadelta = opt.Adadelta(mnl_model.params)
    assert repr(adadelta) == "Adadelta"
    updates = adadelta.update(mnl_model.cost, mnl_model.params)
    assert len(updates) == 12


def test_adam(mnl_model):
    adam = opt.Adam(mnl_model.params)
    assert repr(adam) == "Adam"
    updates = adam.update(mnl_model.cost, mnl_model.params)
    assert len(updates) == 13


def test_adamax(mnl_model):
    adamax = opt.Adamax(mnl_model.params)
    assert repr(adamax) == "Adamax"
    updates = adamax.update(mnl_model.cost, mnl_model.params)
    assert len(updates) == 13
