import numpy as np
import pandas as pd
import pytest

from pycmtensor.dataset import Dataset
from pycmtensor.expressions import Beta
from pycmtensor.models import MNL, compute


@pytest.fixture
def lpmc_ds():
    df = pd.read_csv("data/lpmc.dat", sep="\t")
    df = df[df["travel_year"] == 2015]
    ds = Dataset(df=df, choice="travel_mode")
    ds.split(0.8)
    return ds


def test_compute(lpmc_ds):
    ds = lpmc_ds
    asc_walk = Beta("asc_walk", 0.0, None, None, 1)
    asc_cycle = Beta("asc_cycle", 0.0, None, None, 0)
    asc_pt = Beta("asc_pt", 0.0, None, None, 0)
    asc_drive = Beta("asc_drive", 0.0, None, None, 0)
    b_time = Beta("b_time", 0.0, None, None, 0)

    U_walk = asc_walk
    U_cycle = asc_cycle
    U_pt = asc_pt
    U_drive = asc_drive + b_time * ds["dur_driving"]

    U = [U_walk, U_cycle, U_pt, U_drive]
    mymodel = MNL(ds, locals(), U)

    compute(
        mymodel,
        ds,
        b_time=-5.009,
        asc_pt=-1.398,
        asc_drive=-4.178,
        asc_cycle=-3.996,
    )
