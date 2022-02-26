#!/usr/bin/env python

import aesara.tensor as aet
import dill as pickle
import pandas as pd
import pytest

import pycmtensor as cmt
from pycmtensor.functions import logit, neg_loglikelihood
from pycmtensor.optimizers import Adam
from pycmtensor.pycmtensor import Beta, PyCMTensorModel, Weights
from pycmtensor.results import Results


class MNLmodel(PyCMTensorModel):
    def __init__(self, db):
        super().__init__(db)
        self.name = "myModel"
        self.inputs = db.tensors()  # keep track of inputs

        # declare model params here
        b_cost = Beta("b_cost", 0.0, None, None, 0)
        b_time = Beta("b_time", 0.0, None, None, 0)
        asc_train = Beta("asc_train", 0.0, None, None, 0)
        asc_car = Beta("asc_car", 0.0, None, None, 0)
        asc_sm = Beta("asc_sm", 0.0, None, None, 1)

        # b_time_s = Beta("b_time_s", 0.0, None, None, 0)

        W1 = Weights("ResNet_01a", (3, 10), 0, True)
        W2 = Weights("ResNet_01b", (10, 3), 0, True)

        # append model params to self.params list
        self.append_to_params(locals())
        # self.append_to_params([b_cost, b_time, asc_train, asc_car, asc_sm, W1, W2])

        # Definition of the utility functions
        # srng = RandomStream(seed=234)
        # rv_n = srng.normal(0, 1, size=(20,))
        # b_time_rnd = b_time + b_time_s * rv_n
        U_1 = b_cost * db["TRAIN_CO"] + b_time * db["TRAIN_TT"] + asc_train
        U_2 = b_cost * db["SM_CO"] + b_time * db["SM_TT"] + asc_sm
        U_3 = b_cost * db["CAR_CO"] + b_time * db["CAR_TT"] + asc_car
        U = [U_1, U_2, U_3]
        # rh = ResLogitLayer(U, W1, W2)

        # definition of the choice output
        self.y = db["CHOICE"]

        # symbolic expression for the choice model
        self.p_y_given_x = logit(U, [db["TRAIN_AV"], db["SM_AV"], db["CAR_AV"]])
        # self.p_y_given_x = logit(rh.output, [db["TRAIN_AV"], db["SM_AV"], db["CAR_AV"]])

        # declare Regularizers here:
        # L1 regularization cost
        self.L1 = abs(b_cost()) + abs(b_time())

        # L2 regularization cost
        self.L2 = b_cost() ** 2 + b_time() ** 2

        # symbolic expression for the cost fuction
        self.cost = neg_loglikelihood(self.p_y_given_x, self.y)
        self.cost = self.cost

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.pred = aet.argmax(self.p_y_given_x, axis=0)


@pytest.fixture
def db(dataset="data/swissmetro.dat"):
    swissmetro = pd.read_csv(dataset, sep="\t")
    db = cmt.Database("swissmetro", swissmetro, choiceVar="CHOICE")
    globals().update(db.variables)
    # Removing some observations
    exclude = ((PURPOSE != 1) * (PURPOSE != 3) + (CHOICE == 0)) > 0
    db.remove(exclude)

    # additional steps to format database
    db.data["CHOICE"] -= 1  # set the first choice to 0
    db.autoscale(
        variables=["TRAIN_CO", "TRAIN_TT", "CAR_CO", "CAR_TT", "SM_CO", "SM_TT"],
        default=100.0,
        verbose=False,
    )

    return db


def test_train_mnl(db):
    model = cmt.train(
        MNLmodel, database=db, optimizer=Adam, batch_size=256, lr_init=0.01, max_epoch=5
    )
    # result = Results(model, db, show_weights=True)
