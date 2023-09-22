import numpy as np
import pytest

from pycmtensor.expressions import Beta, Bias, Weight
from pycmtensor.regularizers import Regularizers


def test_l1():
    rng = np.random.default_rng(123)
    beta1 = Beta("beta1", 0.2)
    weight1 = Weight("weight1", (5, 5))
    bias1 = Bias("bias1", (5,), value=rng.uniform(size=(5,)))

    params = [beta1, weight1, bias1]
    l1_reg = Regularizers.l1(beta1)
    l1_reg = Regularizers.l1(params).eval()

    assert len(l1_reg.shape) == 0
    assert np.round(l1_reg, 3) == 0.003


def test_l2():
    rng = np.random.default_rng(123)
    beta1 = Beta("beta1", 0.2)
    weight1 = Weight("weight1", (5, 5))
    bias1 = Bias("bias1", (5,), value=rng.uniform(size=(5,)))

    params = [beta1, weight1, bias1]
    l2_reg = Regularizers.l2(beta1)
    l2_reg = Regularizers.l2(params).eval()

    assert len(l2_reg.shape) == 0
    assert np.round(l2_reg, 4) == 0.0001
