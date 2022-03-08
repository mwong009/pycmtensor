import pytest

from pycmtensor.configparser import config


def test_generate_config_file():
    config
    config.set_num_threads()
    config.generate_config_file()

    # check if at least one blas flag found
    assert len(config["BLAS_FLAGS"]) > 0
