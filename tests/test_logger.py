import pytest

import pycmtensor.logger as log


def test_default_logger():
    log.info("this is an info")
    log.set_level(log.CRITICAL)
