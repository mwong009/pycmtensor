#!/usr/bin/env python

"""Tests for `pycmtensor` package."""
import unittest

import aesara.tensor as aet
import dill as pickle
import numpy as np
import pandas as pd

import pycmtensor as cmt
from pycmtensor.functions import logit, neg_loglikelihood
from pycmtensor.optimizers import Adam
from pycmtensor.pycmtensor import Beta, PyCMTensorModel, Weights, build_functions
from pycmtensor.results import Results


class MNLModel(PyCMTensorModel):
    def __init__(self, db):
        super().__init__()
        self.inputs = db.tensors()  # keep track of inputs
        for var in self.inputs:
            globals().update({var.name: var})

        b_cost = Beta("b_cost", 0.0, None, None, 0)
        b_time = Beta("b_time", 0.0, None, None, 0)
        asc_train = Beta("asc_train", 0.0, None, None, 0)
        asc_car = Beta("asc_car", 0.0, None, None, 0)
        asc_sm = Beta("asc_sm", 0.0, None, None, 1)

        W1 = Weights("ResNet_01a", (3, 10), 0, True)
        W2 = Weights("ResNet_01b", (10, 3), 0, True)

        self.append_to_params([b_cost, b_time, asc_train, asc_car, asc_sm, W1, W2])
        U_1 = b_cost * TRAIN_CO + b_time * TRAIN_TT + asc_train
        U_2 = b_cost * SM_CO + b_time * SM_TT + asc_sm
        U_3 = b_cost * CAR_CO + b_time * CAR_TT + asc_car
        U = [U_1, U_2, U_3]
        self.y = CHOICE
        self.p_y_given_x = logit(U, [TRAIN_AV, SM_AV, CAR_AV])
        self.cost = neg_loglikelihood(self.p_y_given_x, self.y)
        self.pred = aet.argmax(self.p_y_given_x, axis=1)


class TestPycmtensor(unittest.TestCase):
    """Tests for `pycmtensor` package."""

    def setUp(self):
        swissmetro = pd.read_csv("data/swissmetro.dat", sep="\t")
        db = cmt.Database("swissmetro", swissmetro, choiceVar="CHOICE")
        exclude = (
            (db.variables["PURPOSE"] != 1) * (db.variables["PURPOSE"] != 3)
            + (db.variables["CHOICE"] == 0)
        ) > 0
        db.remove(exclude)

        # additional steps to format database
        db.data["CHOICE"] -= 1  # set the first choice to 0
        db.autoscale(
            variables=["TRAIN_CO", "TRAIN_TT", "CAR_CO", "CAR_TT", "SM_CO", "SM_TT"],
            default=100.0,
            verbose=False,
        )
        self.db = db

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_functions(self):
        optimizer = Adam

        print("Building model...")
        self.model = build_functions(MNLModel(self.db), optimizer)

        self.model.null_ll = self.model.loglikelihood(*self.db.input_data())
        self.model.null_score = 1 - self.model.output_errors(*self.db.input_data())
        self.model.probabilities = self.model.output_probabilities(
            *self.db.input_data()
        )
        self.model.best_params = self.model.output_estimated_betas()
        self.model.best_weights = self.model.output_estimated_weights()
        self.model.best_ll = self.model.null_ll

    def test_learning_rate_tempering(self):
        from pycmtensor.utils import learn_rate_tempering

        for iter, patience, f in [[50, 500, 1], [30, 100, 0.2], [70, 100, 0.1]]:
            lr = learn_rate_tempering(iter, patience, lr_init=0.01)
            assert lr == 0.01 * f
