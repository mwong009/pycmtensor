# conftest.py
import pandas as pd
import pytest

import pycmtensor as cmt
from pycmtensor.config import Config
from pycmtensor.expressions import Beta
from pycmtensor.models.MNL import MNL


@pytest.fixture(scope="session")
def config_class():
    return Config()


@pytest.fixture(scope="session")
def swissmetro_db():
    assert cmt.logger.get_effective_level() == 20

    swissmetro = pd.read_csv("data/swissmetro.dat", sep="\t")
    swissmetro.drop(swissmetro[swissmetro["CHOICE"] == 0].index, inplace=True)
    swissmetro["CHOICE"] -= 1  # set the first choice index to 0

    db = cmt.Data(df=swissmetro, choice="CHOICE")
    db.autoscale_data(except_for=["ID", "ORIGIN", "DEST"])  # scales dataset
    db.split_db(split_frac=0.8)  # split dataset
    return db


@pytest.fixture(scope="session")
def betas():
    b_cost = Beta("b_cost", 0.0, None, None, 0)
    b_time = Beta("b_time", 0.0, None, None, 0)
    asc_train = Beta("asc_train", 0.0, None, None, 0)
    asc_car = Beta("asc_car", 0.0, None, None, 0)
    asc_sm = Beta("asc_sm", 0.0, None, None, 1)
    return {
        "b_cost": b_cost,
        "b_time": b_time,
        "asc_train": asc_train,
        "asc_car": asc_car,
        "asc_sm": asc_sm,
    }


@pytest.fixture(scope="session")
def utility(swissmetro_db, betas):
    db = swissmetro_db
    globals().update(betas)
    U_1 = b_cost * db["TRAIN_CO"] + b_time * db["TRAIN_TT"] + asc_train
    U_2 = b_cost * db["SM_CO"] + b_time * db["SM_TT"] + asc_sm
    U_3 = b_cost * db["CAR_CO"] + b_time * db["CAR_TT"] + asc_car

    U = [U_1, U_2, U_3]  # utility
    return U


@pytest.fixture(scope="session")
def availability(swissmetro_db):
    db = swissmetro_db
    AV = [db["TRAIN_AV"], db["SM_AV"], db["CAR_AV"]]  # availability
    return AV


@pytest.fixture(scope="session")
def mnl_model(utility, availability, betas, swissmetro_db):
    return MNL(utility, availability, betas, swissmetro_db)
