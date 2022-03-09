import sys

import pytest

from pycmtensor.config import Config


def test_generate_config_file():
    config = Config()

    # check if config can be accessed
    assert config["python_version"] == sys.version
    print(config)
