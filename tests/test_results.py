import dill as pickle
import pandas as pd
import pytest

import pycmtensor as cmt

# from pycmtensor.pycmtensor import PyCMTensorModel
from pycmtensor.results import Predict, Results, time_format


@pytest.fixture
def test_db():
    swissmetro = pd.read_csv("data/swissmetro.dat", sep="\t")
    db = cmt.Database(name="swissmetro", pandasDatabase=swissmetro, choiceVar="CHOICE")
    globals().update(db.variables)
    # Removing some observations
    # exclude = ((PURPOSE != 1) * (PURPOSE != 3) + (CHOICE == 0)) > 0
    # db.remove(exclude)
    db.data.drop(db.data[db.data["CHOICE"] == 0].index, inplace=True)

    # additional steps to format database
    db.data["CHOICE"] -= 1  # set the first choice to 0
    db.choices = sorted(db.data["CHOICE"].unique())  # save original choices
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


# def test_Results(test_db):
#     with open("tests/model.pkl", "rb") as f:
#         model = pickle.load(f)

#     r = Results(model, test_db)
#     print(r)
#     assert r.n_params == 4

#     r.print_beta_statistics()
#     r.print_correlation_matrix()
#     assert r.print_nn_weights() == None


# def test_Predict(test_db):
#     with open("tests/model.pkl", "rb") as f:
#         model = pickle.load(f)
#     p = Predict(model, test_db)
#     assert len(p.probs()) == len(test_db.data)
#     assert len(p.probs().columns) == 3
#     assert p.choices().shape == (len(test_db.data), 1)
