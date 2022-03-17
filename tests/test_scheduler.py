import pytest

import pycmtensor.scheduler as schlr


@pytest.fixture
def class_CyclicLR():
    return schlr.CyclicLR(base_lr=0.01, max_lr=0.1)


@pytest.fixture
def class_ConstantLR():
    return schlr.ConstantLR(base_lr=0.01)


def test_cyclicLR(class_CyclicLR):
    x = 0.1
    assert class_CyclicLR._triangular_scale_fn(x) == 1
    assert class_CyclicLR.get_lr(0) == 0.01
    assert class_CyclicLR.get_lr(8) == 0.1
    assert class_CyclicLR.get_lr(16) == 0.01


def test_ConstantLR(class_ConstantLR):
    assert class_ConstantLR.get_lr() == 0.01


def test_invalid_mode():
    with pytest.raises(ValueError):
        schlr.CyclicLR(base_lr=0.01, max_lr=0.1, mode="invalid")
