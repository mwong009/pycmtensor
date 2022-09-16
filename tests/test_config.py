# test_config.py
import configparser
import os

import pytest


@pytest.fixture(scope="module")
def config_file():
    HOMEPATH = os.path.expanduser("~")
    conf_file = os.path.join(HOMEPATH, ".aesararc")
    c = configparser.ConfigParser()
    c.read(conf_file)
    return c


def test_aesara_rc(config_class, config_file):
    assert hasattr(config_class, "aesara_rc")
    assert "blas" in config_file.sections()
    assert config_file["global"]["floatX"] == "float64"


def test_set_hyperparameters(config_class):
    print(config_class)
    config_class.set_hyperparameter("max_steps", 100)
    assert config_class.hyperparameters["max_steps"] == 100


def test_call_magic(config_class):
    seed = config_class["seed"]
    assert config_class()["seed"] == seed


def test_setitem_magic(config_class):
    config_class["base_learning_rate"] = 0.02
    with pytest.raises(NameError):
        config_class["learning_rate"] = 0.01


def test_getitem_magic(config_class):
    assert config_class["patience_increase"] == 2
    assert config_class["learning_scheduler"] == "ConstantLR"
