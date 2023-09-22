import numpy as np
import pandas as pd
import pytest

from pycmtensor.dataset import Dataset
from pycmtensor.models.layers import DenseLayer, TNBetaLayer


@pytest.fixture
def lpmc_ds():
    df = pd.read_csv("data/lpmc.dat", sep="\t")
    df = df[df["travel_year"] == 2015]
    ds = Dataset(df=df, choice="travel_mode")
    ds.split(0.8)
    return ds


def test_dense_layer(lpmc_ds):
    ds = lpmc_ds
    x = ds["age", "female", "faretype", "distance", "day_of_week"]

    with pytest.raises(KeyError):
        hl_tanh = DenseLayer(
            name="tn_l1",
            input=x,
            n_in=5,
            n_out=5,
            init_type="invalid_type",
            activation="sigmoid",
        )

    for activation in ["tanh", "relu", "softplus", "sigmoid", None]:
        n_out = np.random.randint(10, 50)
        hl_tanh = DenseLayer(
            name="tn_l1",
            input=x,
            n_in=5,
            n_out=n_out,
            init_type="glorot",
            activation=activation,
        )
        repr(hl_tanh)
        assert hl_tanh.output == hl_tanh()
        assert hl_tanh.name == "tn_l1"

    tnb_layer = TNBetaLayer(name="tnb_layer", input=hl_tanh, lb=-10, ub=10, status=1)
    repr(tnb_layer)
    assert tnb_layer.output == tnb_layer()
    assert tnb_layer.name == "tnb_layer"
    assert isinstance(tnb_layer.get_value(), dict)

    with pytest.raises(TypeError):
        tnb_layer = TNBetaLayer(name="tnb_layer", input=hl_tanh.output)
