# test_scheduler.py
import copy
from collections import OrderedDict

import pytest

from pycmtensor.scheduler import *


def test_set_lr_scheduler(config_class):
    config = copy.copy(config_class)
    config.set_lr_scheduler(ConstantLR())

    assert config["lr_scheduler"].name == "ConstantLR"

    str(config["lr_scheduler"])
    repr(config["lr_scheduler"])

    config.set_lr_scheduler(CyclicLR())

    assert config["lr_scheduler"].name == "CyclicLR"
    assert config["max_learning_rate"] == 0.1
    assert config["clr_cycle_steps"] == 16

    config.set_lr_scheduler(ExpRangeCLR())

    assert config["clr_gamma"] == 0.5


def test_constantlr():
    lr_scheduler = ConstantLR(lr=0.02)
    assert lr_scheduler.lr == 0.02
    assert isinstance(lr_scheduler.history, OrderedDict)

    lr = lr_scheduler(1)
    assert lr == 0.02
    assert len(lr_scheduler.history) == 1


def test_steplr():
    with pytest.raises(ValueError):
        lr_scheduler = StepLR(lr=0.02, factor=1.2, drop_every=10)
    with pytest.raises(ValueError):
        lr_scheduler = StepLR(lr=0.02, factor=0.25, drop_every=1.2)

    lr_scheduler = StepLR(lr=0.1, factor=0.5, drop_every=5)
    assert lr_scheduler.lr == 0.1
    assert lr_scheduler.factor == 0.5
    assert lr_scheduler.drop_every == 5

    lr = [lr_scheduler(i) for i in range(15)]
    assert lr[-1] == 0.025
    assert len(lr_scheduler.history) == 15


def test_polynomiallr():
    with pytest.raises(ValueError):
        lr_scheduler = PolynomialLR(200, lr=0.1, power=-0.5)

    lr_scheduler = PolynomialLR(200, lr=0.1, power=2.0)
    assert lr_scheduler.power == 2.0
    assert lr_scheduler.max_steps == 200

    lr = [lr_scheduler(i) for i in range(200)]
    assert lr[-1] <= 1e-4
    assert len(lr_scheduler.history) == 200


def test_cycliclr():
    with pytest.raises(ValueError):
        lr_scheduler = CyclicLR(lr=0.1, max_lr=0.05, cycle_steps=12)

    lr_scheduler = CyclicLR(lr=0.01, max_lr=0.5, cycle_steps=16)
    assert lr_scheduler.max_lr == 0.5
    assert lr_scheduler.cycle_steps == 16

    lr = [lr_scheduler(i) for i in range(100)]
    assert lr[16] == 0.01
    assert lr[8] == 0.5
    assert len(lr_scheduler.history) == 100


def test_triangular2clr():
    lr_scheduler = Triangular2CLR(lr=0.01, max_lr=0.5, cycle_steps=16)
    assert lr_scheduler.scale_fn(1) == 1.0
    assert lr_scheduler.scale_fn(2) == 0.5
    assert lr_scheduler.scale_fn(3) == 0.25


def test_exprangeclr():
    with pytest.raises(ValueError):
        lr_scheduler = ExpRangeCLR(gamma=1.9)

    lr_scheduler = ExpRangeCLR(lr=0.01, max_lr=0.5, cycle_steps=16, gamma=0.9)

    assert lr_scheduler.gamma == 0.9
    assert lr_scheduler.scale_fn(1) == 0.9
    assert lr_scheduler.scale_fn(2) == 0.9**2
    assert lr_scheduler.scale_fn(3) == 0.9**3
