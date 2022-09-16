# test_expressions.py
import pytest
from aesara import function
from aesara.tensor.sharedvar import TensorSharedVariable

import pycmtensor.expressions as expressions


@pytest.fixture(scope="module")
def exp_parser():
    return expressions.ExpressionParser()


@pytest.fixture
def beta_class():
    return expressions.Beta("b_cost", 1.0, -10.0, 10, 0)


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


def test_beta_reset(beta_class):
    b_cost = beta_class
    b_cost.init_value = 5.0
    assert b_cost.get_value() == 5.0


def test_beta_update(beta_class):
    b_cost = beta_class
    assert b_cost.get_value() == 1.0
    f = function(
        inputs=[],
        outputs=b_cost(),
        updates=[(b_cost(), b_cost() + 1.0)],
    )
    for _ in range(2):
        value = f()
    assert value == 2.0
