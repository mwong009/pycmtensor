import aesara.tensor as aet
import numpy as np
import pytest
from aesara import function, shared

from pycmtensor.expressions import Beta
from pycmtensor.optimizers import SGD


def test_SGD():
    x = aet.scalar("x")
    y = aet.scalar("y")
    beta = Beta("beta", 2.0, None, None, 0)
    opt = SGD()
    y = beta.sharedVar * x + 3
    updates = opt.update(y, [beta], lr=1.0)
    f = function([x], updates[0][1], updates=updates)
    b = function([x], y)
    # dy/dbeta = x
    # beta = beta - x = 2 - 4 = -2
    # y = -2 * 4 + 3 = -5
    assert f(4) == -2
    assert b(4) == -5
