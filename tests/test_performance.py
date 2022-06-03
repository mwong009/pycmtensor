import sys

import pytest

from pycmtensor.expressions import Beta


@pytest.fixture
def database():
    import pandas as pd

    from pycmtensor import Database

    sm = pd.read_csv("data/swissmetro.dat", sep="\t")
    db = Database(name="swissmetro", pandasDatabase=sm, choiceVar="CHOICE")
    globals().update(db.variables)
    # exclude = ((PURPOSE != 1) * (PURPOSE != 3) + (CHOICE == 0)) > 0
    # db.remove(exclude)
    db.data.drop(db.data[db.data["CHOICE"] == 0].index, inplace=True)
    db.data["CHOICE"] -= 1  # set the first choice to 0
    db.choices = sorted(db.data["CHOICE"].unique())  # save original choices
    db.autoscale(
        variables=["TRAIN_CO", "TRAIN_TT", "CAR_CO", "CAR_TT", "SM_CO", "SM_TT"],
        default=100.0,
        verbose=False,
    )
    return db


def test_performace_model(database):
    print("\nPerformance Tests")
    import pycmtensor as cmt
    from pycmtensor.models import MNLogit
    from pycmtensor.optimizers import Adam
    from pycmtensor.results import Results

    db = database
    cmt.logger.set_level(cmt.logger.ERROR)
    b_cost = Beta("b_cost", 0.0, None, None, 0)
    b_time = Beta("b_time", 0.0, None, None, 0)
    U_1 = b_cost * db["TRAIN_CO"] + b_time * db["TRAIN_TT"]
    U_2 = b_cost * db["SM_CO"] + b_time * db["SM_TT"]
    U_3 = b_cost * db["CAR_CO"] + b_time * db["CAR_TT"]
    U = [U_1, U_2, U_3]
    AV = [db["TRAIN_AV"], db["SM_AV"], db["CAR_AV"]]
    model = MNLogit(u=U, av=AV, database=db, name="model")
    model.add_params(locals())
    model.config["verbosity"] = "low"
    model.config["max_epoch"] = 100
    model.config["base_lr"] = 0.0012
    model.config["learning_scheduler"] = "ConstantLR"

    model.config["debug"] = True
    train_model = cmt.train(model, database=db, optimizer=Adam, batch_size=128)
    print(f"Debug estimation rate: {train_model.iter_per_sec} iter/s\n")

    model.config["debug"] = False
    train_model = cmt.train(model, database=db, optimizer=Adam, batch_size=128)
    print(f"Notebook estimation rate: {train_model.iter_per_sec} iter/s\n")
