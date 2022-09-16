# test_logger.py
import pytest

import pycmtensor.logger as logger


@pytest.fixture(scope="module")
def main_logger():
    return logger.main_logger


def test_set_level():
    logger.set_level(logger.DEBUG)
    assert logger.get_effective_level() == 10


def test_logging(main_logger):
    main_logger.log(level=logger.INFO, msg=f"Hello world")
