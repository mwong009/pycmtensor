import copy

import numpy as np
import pandas as pd
import pytest
from aesara.tensor.var import TensorVariable

from pycmtensor.dataset import Dataset


@pytest.fixture
def df():
    df = pd.read_csv("data/lpmc.dat", sep="\t")
    df = df[df["travel_year"] == 2015]
    return df


@pytest.fixture
def lpmc_ds(df):
    ds = Dataset(df=df, choice="travel_mode")
    ds.split(0.8)
    return ds


def test_add_config_value(df):
    Dataset(df=df, choice="travel_mode", batch_size=32)


def test_choice_error(df):
    with pytest.raises(IndexError):
        Dataset(df=df, choice="not_travel_mode")
    ds = Dataset(df=df, choice="travel_mode")
    with pytest.raises(KeyError):
        ds["not_travel_mode"]


def test_split_count(df):
    ds = Dataset(df=df, choice="travel_mode")
    ds.split(frac=None, count=100)
    assert len(ds.idx_train) == 100
    assert len(ds.idx_valid) == len(df) - 100


def test_split_frac_one(df):
    ds = Dataset(df=df, choice="travel_mode")

    len_dataset = len(ds.ds["travel_mode"])
    assert ds.split_frac == 1
    assert len(ds.idx_train) == len_dataset
    assert len(ds.idx_valid) == len_dataset


def test_dataset_property(lpmc_ds):
    ds = lpmc_ds.ds

    len_dataset = len(ds["travel_mode"])
    frac = lpmc_ds.split_frac
    len_train_dataset = round(len_dataset * frac)
    len_valid_dataset = len_dataset - round(len_dataset * frac)

    assert lpmc_ds.n_train == len_train_dataset
    assert lpmc_ds.n_valid == len_valid_dataset

    assert all(lpmc_ds.idx_train == lpmc_ds.index[:len_train_dataset])
    assert all(lpmc_ds.idx_valid == lpmc_ds.index[len_train_dataset:])

    assert ds == lpmc_ds()

    assert lpmc_ds["travel_mode"].name == "travel_mode"
    assert isinstance(lpmc_ds["travel_mode"], TensorVariable)

    assert lpmc_ds["travel_year"].name == "travel_year"
    assert isinstance(lpmc_ds["travel_year"], TensorVariable)

    with pytest.raises(KeyError):
        lpmc_ds["unknow_var"]


def test_train_dataset(lpmc_ds):
    ds = lpmc_ds
    set1 = ds.train_dataset(["dur_walking"])
    set1a = ds.train_dataset(ds["dur_walking"])
    set2 = ds.train_dataset(["dur_walking", "dur_driving", "dur_pt_rail"])
    set3 = ds.train_dataset([ds["dur_walking"], ds["dur_driving"], ds["dur_pt_rail"]])

    for i, j in zip(set1, set1a):
        assert all(i == j)
    for i, j in zip(set2, set3):
        assert all(i == j)

    set4 = ds.train_dataset(["cost_transit"], index=5, batch_size=32)
    set4a = ds.train_dataset(["cost_transit"], index=5)

    assert len(set4[0]) == 32
    assert len(set4a[0]) == 1

    with pytest.raises(TypeError):
        set5 = ds.train_dataset([ds["dur_walking"], "dur_driving", "dur_pt_rail"])

    with pytest.raises(KeyError):
        set6 = ds.train_dataset(ds["not_dur_walking"])

    with pytest.raises(KeyError):
        set7 = ds.train_dataset(["not_dur_walking"])


def test_valid_dataset(lpmc_ds):
    ds = lpmc_ds
    set1 = ds.valid_dataset(["dur_walking"])


def test_drop(lpmc_ds):
    lpmc_ds_cp = copy.copy(lpmc_ds)

    assert "distance" in lpmc_ds_cp.scale
    assert "distance" in lpmc_ds_cp.ds
    assert "distance" in [var.name for var in lpmc_ds_cp.x]

    lpmc_ds_cp.drop(["distance"])

    assert not "distance" in lpmc_ds_cp.scale
    assert not "distance" in lpmc_ds_cp.ds
    assert not "distance" in [var.name for var in lpmc_ds_cp.x]

    with pytest.raises(KeyError):
        lpmc_ds_cp.drop("distance")


def test_scale_variable(lpmc_ds):
    ds = copy.copy(lpmc_ds.ds)

    # test single variable
    lpmc_ds.scale_variable("distance", 1000.0)
    assert lpmc_ds.scale["distance"] == 1000.0
    assert all(lpmc_ds.ds["distance"] == ds["distance"] / 1000.0)
    lpmc_ds.scale_variable("distance", 1 / 1000.0)

    # test multiple variables
    lpmc_ds.scale_variable(["distance", "cost_transit"], 10.0)
    assert lpmc_ds.scale["distance"] == 10.0
    assert lpmc_ds.scale["cost_transit"] == 10.0
    assert np.abs(np.sum(lpmc_ds.ds["distance"] - ds["distance"] / 10.0)) < 1e-5
    assert np.sum(lpmc_ds.ds["cost_transit"] - ds["cost_transit"] / 10.0) == 0.0
    lpmc_ds.scale_variable(["distance", "cost_transit"], 1 / 10.0)

    # test non-string
    with pytest.raises(TypeError):
        lpmc_ds.scale_variable(123, 10.0)


def test_make_tensor(lpmc_ds):
    import aesara.tensor as aet

    ds = lpmc_ds
    tensors1 = ds[["cost_transit", "cost_driving_ccharge"]]
    tensors2 = aet.as_tensor_variable([ds["cost_transit"], ds["cost_driving_ccharge"]])

    assert tensors1.ndim == tensors2.ndim

    with pytest.raises(TypeError):
        tensors3 = ds[[ds["cost_transit"], "cost_driving_ccharge"]]
