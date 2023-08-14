# test_expressions.py
from copy import copy

import aesara
import aesara.tensor as aet
import numpy as np
import pytest
from aesara import function
from aesara.tensor.math import tanh
from aesara.tensor.sharedvar import TensorSharedVariable

from pycmtensor.expressions import (
    Beta,
    Bias,
    ExpressionParser,
    Param,
    RandomDraws,
    TensorExpressions,
    Weight,
)


@pytest.fixture(scope="module")
def exp_parser():
    # random equation
    x = aet.scalar("x")
    b = aet.scalar("b")
    y = 3 * x + b
    ep = ExpressionParser(y)
    assert isinstance(ep.expression, str)

    return ExpressionParser()


def test_parser():
    expression = "(-(sum(AdvancedSubtensor(log(((Softmax{axis=0}(Reshape{2}(join(0, (((b_cost * TRAIN_CO) + (b_time * TRAIN_TT)) + asc_train), (((b_cost * SM_CO) + (b_time * SM_TT)) + asc_sm), (((b_cost * CAR_CO) + (b_time * CAR_TT)) + asc_car)), [Shape(join(0, (((b_cost * TRAIN_CO) + (b_time * TRAIN_TT)) + asc_train), (((b_cost * SM_CO) + (b_time * SM_TT)) + asc_sm), (((b_cost * CAR_CO) + (b_time * CAR_TT)) + asc_car)))[:int64][0], int64(-1)])) * join(0, TRAIN_AV, SM_AV, CAR_AV)) / sum((Softmax{axis=0}(Reshape{2}(join(0, (((b_cost * TRAIN_CO) + (b_time * TRAIN_TT)) + asc_train), (((b_cost * SM_CO) + (b_time * SM_TT)) + asc_sm), (((b_cost * CAR_CO) + (b_time * CAR_TT)) + asc_car)), [Shape(join(0, (((b_cost * TRAIN_CO) + (b_time * TRAIN_TT)) + asc_train), (((b_cost * SM_CO) + (b_time * SM_TT)) + asc_sm), (((b_cost * CAR_CO) + (b_time * CAR_TT)) + asc_car)))[:int64][0], int64(-1)])) * join(0, TRAIN_AV, SM_AV, CAR_AV)), axis=(0,)))), CHOICE, ARange{dtype='int64'}(0, Shape(CHOICE)[0], 1)), axis=None) / Shape(CHOICE)[0]))"
    symbols = ExpressionParser.parse(expression)
    variables = ["SM_TT", "SM_CO", "CAR_TT", "CAR_CO", "TRAIN_TT", "TRAIN_CO"]
    assert all([s in symbols for s in variables])


def test_construct_beta():
    b_test = Beta("b_test", value=1.0)

    assert isinstance(b_test, Param)
    assert isinstance(b_test, TensorExpressions)

    with pytest.raises(NotImplementedError):
        b_test.T

    with pytest.raises(NotImplementedError):
        assert b_test.init_type == 2


def test_construct_weights_and_biases():
    w_layer1 = Weight("w_layer1", size=[3, 5])
    bias_layer1 = Bias("bias_layer1", size=(5,))

    assert bias_layer1.init_type == "bias"
    assert all(bias_layer1.T.eval() == bias_layer1.get_value())

    assert w_layer1.init_type == "uniform(-0.1, 0.1)"

    w_layer1 = Weight("w_layer1", size=[3, 5], init_type="zeros")
    w_layer1 = Weight("w_layer1", size=[3, 5], init_type="glorot")
    w_layer1 = Weight("w_layer1", size=[3, 5], init_type="he")

    with pytest.raises(ValueError):
        w_layer1_fail = Weight("w_layer1_fail", size=[3, 5, 4])

    with pytest.raises(ValueError):
        bias_layer1_fail = Bias("bias_layer1_fail", size=[3, 4])

    with pytest.raises(TypeError):
        bias_layer1_fail = Bias("bias_layer1_fail", size=4)


def test_bias_add():
    x = aet.vector("x")
    bias_layer1 = Bias("bias_layer1", size=(5,))

    g = aesara.shared(np.random.normal(size=(3, 5)))
    y = bias_layer1 + g

    assert (aet.eq(g.shape[-1], bias_layer1.shape[0])).eval()
    assert y.eval().shape == (3, 5)

    w_layer1 = Weight("w_layer1", size=[3, 5], init_type="he")
    y = (bias_layer1 + w_layer1) + (w_layer1 + bias_layer1)
    assert y.eval().shape == (3, 5)


def test_construct_random_draws():
    rnd_test = RandomDraws("rnd_test", "normal", 20)
    rnd_test_log = RandomDraws("rnd_test_log", "lognormal", 20)
    gumbel = RandomDraws("rnd_test2", "gumbel", 20)
    exponential = RandomDraws("rnd_test2", "exponential", 20)
    gamma = RandomDraws("rnd_test2", "gamma", 20)
    poisson = RandomDraws("rnd_test2", "poisson", 20)

    with pytest.raises(NotImplementedError):
        rnd_test_3 = RandomDraws("rnd_test3", "notgumbel", 20)
