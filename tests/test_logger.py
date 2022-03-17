import pytest

import pycmtensor.logger as log


def test_default_logger():
    log.info("this is an info")
    log.set_level(log.CRITICAL)


def test_debug_logger():
    debug_logger = log.get_debug_logger()
    debug_logger.info("this is an info")
