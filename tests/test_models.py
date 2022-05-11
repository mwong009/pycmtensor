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


@pytest.fixture
def utility(database):
    db = database
    b_cost = Beta("b_cost", 0.0, None, None, 0)
    b_time = Beta("b_time", 0.0, None, None, 0)
    U_1 = b_cost * db["TRAIN_CO"] + b_time * db["TRAIN_TT"]
    U_2 = b_cost * db["SM_CO"] + b_time * db["SM_TT"]
    U_3 = b_cost * db["CAR_CO"] + b_time * db["CAR_TT"]
    U = [U_1, U_2, U_3]
    return U, [b_cost, b_time]


def test_detect_duplicate_params(database):
    from pycmtensor.models import MNLogit

    db = database
    b_cost = Beta("b_cost", 0.0, None, None, 0)
    b_time = Beta("b_cost", 0.0, None, None, 0)
    U_1 = b_cost * db["TRAIN_CO"] + b_time * db["TRAIN_TT"]
    U_2 = b_cost * db["SM_CO"] + b_time * db["SM_TT"]
    U_3 = b_cost * db["CAR_CO"] + b_time * db["CAR_TT"]
    U = [U_1, U_2, U_3]
    AV = [db["TRAIN_AV"], db["SM_AV"], db["CAR_AV"]]
    model = MNLogit(u=U, av=AV, database=db, name="model")
    with pytest.raises(NameError):
        model.add_params(locals())


def test_incorrect_param_type(utility, database):
    from pycmtensor.models import MNLogit

    db = database
    U, params = utility
    AV = [db["TRAIN_AV"], db["SM_AV"], db["CAR_AV"]]
    model = MNLogit(u=U, av=AV, database=db, name="model")
    with pytest.raises(TypeError):
        model.add_params(params[0])
