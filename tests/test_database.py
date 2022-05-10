import aesara.tensor as aet
import pandas as pd
import pytest

import pycmtensor as cmt


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


def test_compile_data(test_db):
    test_db.compile_data()
    assert test_db.sharedData["CHOICE"].dtype == "int32"


def test_get_rows(test_db):
    assert test_db.get_rows() == 10719


def test_get_tensors(test_db):
    for tensor in test_db.get_tensors():
        if str(tensor) not in test_db.data.columns:
            raise ValueError


def test_input_data(test_db):
    x = aet.vector("TRAIN_TT")
    x_data = test_db.input_data([x])
    assert len(x_data) == 1
    assert len(x_data[0]) == 10719

    all_data = test_db.input_data()
    assert len(all_data) == len(test_db.data.columns)

    slice_data = test_db.input_data([x], index=0, batch_size=32, shift=1)
    assert len(slice_data[0]) == 32


def test_input_shared_data(test_db):
    test_db.compile_data()
    x = aet.vector("TRAIN_TT")
    x_data = test_db.input_shared_data([x])
    assert len(x_data) == 1

    all_data = test_db.input_shared_data()
    assert len(all_data) == len(test_db.data.columns)


def test_autoscale(test_db):
    test_db.autoscale()
    test_db.autoscale(variables=["TRAIN_CO"], default=100, verbose=True)


def test_variable_not_found(test_db):
    with pytest.raises(KeyError):
        abc = test_db["abc"]
        print(abc)
    choice = test_db["CHOICE"]
    train_co = test_db["TRAIN_CO"]
    print(choice, train_co)
