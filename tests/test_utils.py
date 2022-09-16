# test_utils.py
import pytest

import pycmtensor.utils


def test_time_format():
    time = pycmtensor.utils.time_format(5555)
    assert time == "01:32:35"
