# test_utils.py
import pytest

import pycmtensor.utils


def test_time_format():
    time = pycmtensor.utils.time_format(5555)
    assert time == "01:32:35"


def test_human_format():
    human_format = pycmtensor.utils.human_format
    assert human_format(1000) == "1K"
    assert human_format(1000000) == "1M"
    assert human_format(1000000000) == "1B"
    assert human_format(1000000000000) == "1T"
    assert human_format(1000000000000000) == "1P"
