# models.py
import timeit

import aesara.tensor as aet
from aesara import function, pprint

from ..functions import bhhh, errors, gnorm, hessians, log_likelihood, logit
from ..logger import log
from ..optimizers import Adam
from ..pycmtensor import PyCMTensorModel
from ..utils import time_format


class MNL(PyCMTensorModel):
    def __init__(self, utility, av, params, db, optimizer=Adam, name="MNL"):
        super().__init__(db)

        self.name = name
        self.learning_rate = aet.scalar("learning_rate")

        # Build model
        log(10, f"Building model...")
        start_time = timeit.default_timer()

        # Definition of the symbolic choice output (tensor)
        self.y = db.tensor.choice

        # symbolic expression for the choice model probability
        self.p_y_given_x = logit(utility, av)

        # symbolic expression for the likelihood
        self.ll = log_likelihood(self.p_y_given_x, self.y)

        # the cost function to minimize is the negative loglikelihood
        self.cost = -(self.ll / self.y.shape[0])

        # symbolic description of how to compute prediction as class whose probability
        # is maximal
        self.pred = aet.argmax(self.p_y_given_x, axis=0)

        # add params to track in self.params and self.betas
        self.add_params(params)

        # define the optimizer for the model
        self.opt = optimizer(self.params)
        updates = self.opt.update(self.cost, self.params, self.learning_rate)

        # define the update function to update the parameters wrt to the cost
        self.update_wrt_cost = function(
            inputs=self.inputs + [self.learning_rate],
            outputs=self.cost,
            updates=updates,
            on_unused_input="ignore",
            name="update_wrt_cost",
        )

        # define the function to output the log likelihood of the model
        self.loglikelihood = function(
            inputs=self.inputs,
            outputs=self.ll,
            on_unused_input="ignore",
            name="loglikelihood",
        )

        # define the function to output the predicted probabilities given inputs
        self.choice_probabilities = function(
            inputs=self.inputs,
            outputs=self.p_y_given_x.T,
            on_unused_input="ignore",
            name="choice_probabilities",
        )

        # define the function to output the discrete choice predictions
        self.choice_predictions = function(
            inputs=self.inputs,
            outputs=self.pred,
            on_unused_input="ignore",
            name="choice_predictions",
        )

        # define the function to output the choice prediction error
        self.prediction_error = function(
            inputs=self.inputs,
            outputs=errors(self.p_y_given_x, self.y),
            on_unused_input="ignore",
            name="errors",
        )

        # define the function to ouput the Hessian matrix or the 2nd-order partial
        # derivatives
        self.H = function(
            inputs=self.inputs,
            outputs=hessians(self.ll, self.betas),
            on_unused_input="ignore",
            name="Hessian matrix",
        )

        # define he Berndt–Hall–Hall–Hausman (BHHH) algorithm output function
        self.BHHH = function(
            inputs=self.inputs,
            outputs=bhhh(self.ll, self.betas),
            on_unused_input="ignore",
            name="BHHH matrix",
        )

        # define the function to output the gradient norm
        self.gradient_norm = function(
            inputs=self.inputs,
            outputs=gnorm(self.cost, self.betas),
            on_unused_input="ignore",
        )

        build_time = round(timeit.default_timer() - start_time, 3)
        self.results.build_time = time_format(build_time)
        log(10, f"Build time = {self.results.build_time}")

        # compute the null loglikelihood
        data = db.pandas.inputs(self.inputs, split_type="train")
        self.results.null_loglikelihood = self.loglikelihood(*data)
        log(10, f"Null loglikelihood = {self.results.null_loglikelihood}")

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return pprint(self.cost)
