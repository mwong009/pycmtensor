# test_data.py
import copy

import numpy as np
import pytest
from aesara.tensor.var import TensorVariable


def test_scale_data(swissmetro_db):
    db = copy.copy(swissmetro_db)
    train_tt = np.mean(db.pandas["TRAIN_TT"])
    db.scale_data(TRAIN_TT=2.0)
    assert np.mean(db.pandas["TRAIN_TT"]) == (train_tt / 2.0)
    assert db.scales["TRAIN_TT"] == 2000


def test_get_nrows(swissmetro_db):
    db = swissmetro_db
    assert db.get_nrows() == len(db.pandas.pandas)


def test_data(swissmetro_db):
    db = swissmetro_db
    assert len(db.x) == 27
    assert type(db.y) == TensorVariable
    assert len(db.all) == 28
    print(db.info())


def test_get_pandas_inputs(swissmetro_db):
    db = swissmetro_db
    assert db.split_frac == 0.8
    train_data = db.pandas.inputs(db.all, split_type="train")
    valid_data = db.pandas.inputs(db.all, split_type="valid")
    assert len(train_data) == 28
    assert len(valid_data) == 28

    n_train_samples = round(db.get_nrows() * db.split_frac)
    for data in train_data:
        assert len(data) == n_train_samples
