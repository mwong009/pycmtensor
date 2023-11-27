import numpy as np
import pandas as pd
import pytest

import pycmtensor.statistics


def test_t_test():
    stderr = pd.Series([1, 2, 3])
    params = {"param1": 4, "param2": 5, "param3": 6}
    expected_result = [4 / 1, 5 / 2, 6 / 3]

    result = pycmtensor.statistics.t_test(stderr, params)
    assert result == expected_result
    assert isinstance(result, list)


def test_stderr():
    hessian = np.array([[2, -1], [-1, 2]])
    params = {"param1": np.array([1]), "param2": np.array([2])}
    float_max = [np.finfo(float).max]

    result = pycmtensor.statistics.stderror(hessian, params)
    assert np.allclose(result, float_max)
    assert isinstance(result, list)

    hessian = np.array([[2, -1], [-1, 2]])
    params = {"param1": np.array(0), "param2": np.array(0)}

    result = pycmtensor.statistics.stderror(hessian, params)
    assert result == [np.nan, np.nan]

    hessian = np.array([[-1.2, -2.3], [-0.2, -4.0]])
    params = {"param1": np.array(1), "param2": np.array(2)}

    result = pycmtensor.statistics.stderror(hessian, params)
    assert np.allclose(result, [0.96, 0.526], rtol=1e-3)


def test_p_value():
    stderr = pd.Series([0.1, 0.2, 0.3])
    params = {"param1": np.array([1, 2, 3]), "param2": np.array([4, 5, 6])}
    expected = [0.0, 0.0]
    assert pycmtensor.statistics.p_value(stderr, params) == expected
