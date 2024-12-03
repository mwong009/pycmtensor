import numpy as np
import pandas as pd
import pytest
from pytensor.tensor.sharedvar import TensorSharedVariable

from pycmtensor.dataset import Dataset
from pycmtensor.expressions import Beta


@pytest.fixture
def lpmc_ds():
    df = pd.read_csv("data/lpmc.dat", sep="\t")
    df = df[df["travel_year"] == 2015]
    ds = Dataset(df=df, choice="travel_mode")
    ds.split(0.8)
    return ds


def test_betas():
    ASC_WALK = Beta("ASC_WALK", 0.0, None, None, 0)
    ASC_CYCLE = Beta("ASC_CYCLE", 0.5, None, None, 0)
    ASC_PT = Beta("ASC_PT", -0.5, None, None, 0)
    ASC_DRIVE = Beta("ASC_DRIVE", 0.0, None, None, 1)

    assert ASC_WALK.name == "ASC_WALK"
    assert ASC_CYCLE.init_value == 0.5
    assert ASC_PT.shape == ()
    assert ASC_DRIVE.status == 1

    assert isinstance(ASC_WALK(), TensorSharedVariable)
    assert ASC_PT.get_value() == np.array(-0.5)

    ASC_DRIVE.set_value(1.99)
    assert ASC_DRIVE.get_value() == np.array(1.99)
    ASC_DRIVE.reset_value()
    assert ASC_DRIVE.get_value() == np.array(0.0)


def test_betas_bounds():
    ASC_WALK = Beta("ASC_WALK", 0.0, -1.0, 1.0, 0)

    assert ASC_WALK.ub == 1
    assert ASC_WALK.lb == -1

    with pytest.raises(ValueError):
        ASC_CYCLE = Beta(f"ASC_CYCLE", 0.5, -1.0, -1.01, 0)

    assert repr(ASC_WALK) == f"Beta(ASC_WALK, 0.0)"


def test_beta_add_subtract(lpmc_ds):
    import pytensor

    ds = lpmc_ds
    B_COST = Beta("B_COST", 1.0, None, None, 0.0)
    x_cost_transit = ds["cost_transit"]

    fn_add = pytensor.function(
        inputs=[x_cost_transit],
        outputs=B_COST + x_cost_transit,
        on_unused_input="ignore",
    )

    train_data = ds.train_dataset(x_cost_transit)
    output = fn_add(*train_data)

    assert (output == (train_data[0] + 1)).all()

    x_cost_driving_fuel = ds["cost_driving_fuel"]

    fn_muladd = pytensor.function(
        inputs=[x_cost_transit, x_cost_driving_fuel],
        outputs=B_COST * (x_cost_driving_fuel + x_cost_transit),
        on_unused_input="ignore",
    )

    train_data = ds.train_dataset([x_cost_transit, x_cost_driving_fuel])
    output = fn_muladd(*train_data)

    assert ((train_data[0] + train_data[1]) == output).all()


def test_beta_multiplication(lpmc_ds):
    import pytensor

    ds = lpmc_ds
    B_COST = Beta("B_COST", 0.0, None, None, 0.0)
    x_cost_transit = ds["cost_transit"]

    fn_mul = pytensor.function(
        inputs=[x_cost_transit],
        outputs=B_COST * x_cost_transit + x_cost_transit * B_COST,
        on_unused_input="ignore",
    )

    train_data = ds.train_dataset(x_cost_transit)

    output = fn_mul(*train_data)

    assert np.sum(output) == 0.0


def test_variable_boolean(lpmc_ds):
    import pytensor
    import pytensor.tensor as aet

    ds = lpmc_ds

    DL = aet.eq(ds["driving_license"], 1)
    data = ds.train_dataset(["driving_license"])

    fn_bool = pytensor.function(inputs=[ds["driving_license"]], outputs=DL)

    output = fn_bool(*data)

    assert (output == data[0]).all()
