from unittest import result

import dill as pickle
import pandas as pd
import pytest
from regex import P

import pycmtensor as cmt
from pycmtensor.expressions import Beta
from pycmtensor.models import MNLogit
from pycmtensor.optimizers import Adam
from pycmtensor.results import Results, time_format


@pytest.fixture
def test_db():
    swissmetro = pd.read_csv("data/swissmetro.dat", sep="\t")
    db = cmt.Database(name="swissmetro", pandasDatabase=swissmetro, choiceVar="CHOICE")
    globals().update(db.variables)
    # Removing some observations
    db.data.drop(db.data[db.data["CHOICE"] == 0].index, inplace=True)
    db.data["CHOICE"] -= 1  # set the first choice index to 0
    db.choices = [0, 1, 2]
    db.autoscale(
        variables=["TRAIN_CO", "TRAIN_TT", "CAR_CO", "CAR_TT", "SM_CO", "SM_TT"],
        default=100.0,
        verbose=False,
    )
    return db


def test_time_format():
    s = 1082
    fmt = time_format(s)
    assert fmt == "00:18:02"


def test_results(test_db):
    with open("tests/model.pkl", "rb") as f:
        model = pickle.load(f)

    results = Results(model, test_db, prnt=False)
    results.generate_beta_statistics()
    results.print_beta_statistics()
    results.print_correlation_matrix()
