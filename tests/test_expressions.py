# test_expressions.py
from copy import copy

import aesara
import aesara.tensor as aet
import numpy as np
import pytest
from aesara import function
from aesara.tensor.math import tanh
from aesara.tensor.sharedvar import TensorSharedVariable

import pycmtensor.expressions as expressions


@pytest.fixture(scope="module")
def exp_parser():
    return expressions.ExpressionParser()


@pytest.fixture
def beta_class():
    return expressions.Beta("b_cost", 1.0, -10.0, 10, 0)


@pytest.fixture
def weight_class(rng):
    return expressions.Weight("w_f1", size=(128, 128), init_type="he", rng=rng)


@pytest.fixture
def rng():
    return np.random.default_rng(42069)


def test_parser(exp_parser):
    expression = "(-(sum(AdvancedSubtensor(log(((Softmax{axis=0}(Reshape{2}(join(0, (((b_cost * TRAIN_CO) + (b_time * TRAIN_TT)) + asc_train), (((b_cost * SM_CO) + (b_time * SM_TT)) + asc_sm), (((b_cost * CAR_CO) + (b_time * CAR_TT)) + asc_car)), [Shape(join(0, (((b_cost * TRAIN_CO) + (b_time * TRAIN_TT)) + asc_train), (((b_cost * SM_CO) + (b_time * SM_TT)) + asc_sm), (((b_cost * CAR_CO) + (b_time * CAR_TT)) + asc_car)))[:int64][0], int64(-1)])) * join(0, TRAIN_AV, SM_AV, CAR_AV)) / sum((Softmax{axis=0}(Reshape{2}(join(0, (((b_cost * TRAIN_CO) + (b_time * TRAIN_TT)) + asc_train), (((b_cost * SM_CO) + (b_time * SM_TT)) + asc_sm), (((b_cost * CAR_CO) + (b_time * CAR_TT)) + asc_car)), [Shape(join(0, (((b_cost * TRAIN_CO) + (b_time * TRAIN_TT)) + asc_train), (((b_cost * SM_CO) + (b_time * SM_TT)) + asc_sm), (((b_cost * CAR_CO) + (b_time * CAR_TT)) + asc_car)))[:int64][0], int64(-1)])) * join(0, TRAIN_AV, SM_AV, CAR_AV)), axis=(0,)))), CHOICE, ARange{dtype='int64'}(0, Shape(CHOICE)[0], 1)), axis=None) / Shape(CHOICE)[0]))"
    symbols = exp_parser.parse(expression)
    assert len(symbols) == 15
    variables = ["SM_TT", "SM_CO", "CAR_TT", "CAR_CO", "TRAIN_TT", "TRAIN_CO"]
    assert all([s in symbols for s in variables])


def test_beta_constructor(beta_class):
    b_cost = beta_class
    assert b_cost.name == "b_cost"
    assert isinstance(b_cost(), TensorSharedVariable)
    with pytest.raises(AttributeError):
        b_cost.status = 1


def test_beta_update(beta_class):
    b_cost = beta_class
    assert b_cost().get_value() == 1.0
    assert b_cost.get_value() == 1.0
    f = function(
        inputs=[],
        outputs=b_cost(),
        updates=[(b_cost(), b_cost() + 1.0)],
    )
    for _ in range(2):
        value = f()
    assert value == 2.0


def test_weight_constructor(weight_class):
    w = weight_class
    assert w.shape == (128, 128)
    assert isinstance(w(), TensorSharedVariable)

    with pytest.raises(ValueError):
        new_weight = expressions.Weight("nw", (2,))

    with pytest.raises(ValueError):
        new_weight = expressions.Weight("nw", (5, 5), init_value=np.eye(3))


def test_weight_init(rng):
    w = expressions.Weight("w_none", (5, 5), rng=rng, init_type=None)
    gl = expressions.Weight("w_glorot", (5, 5), rng=rng, init_type="glorot")


def test_weight_he(weight_class, rng):
    a = aesara.shared(rng.normal(size=(128,)))

    for _ in range(22):
        w = copy(weight_class)
        a = aet.nnet.relu(aet.dot(w(), a))

    assert round(float(aet.mean(a).eval()), 3) == 2.082
    assert round(float(aet.std(a).eval()), 3) == 2.942


def test_weight_glorot(rng):
    glorot = expressions.Weight("w_glorot", (128, 128), rng=rng, init_type="glorot")
    a = aesara.shared(rng.normal(size=(128,)))

    for _ in range(22):
        w = copy(glorot)
        a = tanh(aet.dot(w(), a))
    assert round(float(aet.mean(a).eval()), 3) == 0.008
    assert round(float(aet.std(a).eval()), 3) == 0.194
