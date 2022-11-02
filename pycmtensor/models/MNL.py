# models.py
from time import perf_counter

import aesara.tensor as aet
from aesara import function, pprint

from ..functions import log_likelihood, logit
from ..logger import debug, log
from ..optimizers import Adam
from ..pycmtensor import PyCMTensorModel
from ..utils import time_format


class MNL(PyCMTensorModel):
    def __init__(self, utility, av, params, db, optimizer=Adam, name="MNL"):
        super().__init__(db)

        self.name = name
        self.learning_rate = aet.scalar("learning_rate")

        # Build model
        start_time = perf_counter()

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
        self.updates += self.opt.update(self.cost, self.params, self.learning_rate)

        # define the update function to update the parameters wrt to the cost
        self.update_wrt_cost = function(
            name="update_wrt_cost",
            inputs=self.inputs + [self.learning_rate],
            outputs=self.cost,
            updates=self.updates,
        )

        # internal functions
        self.model_loglikelihood()
        self.model_choice_probabilities()
        self.model_choice_predictions()
        self.model_prediction_error()
        self.model_H()
        self.model_BHHH()
        self.model_gnorm()

        build_time = round(perf_counter() - start_time, 3)
        self.results.build_time = time_format(build_time)
        debug(f"Build time = {self.results.build_time}")

        # compute the null loglikelihood
        data = db.pandas.inputs(self.inputs, split_type="train")
        self.results.null_loglikelihood = self.loglikelihood(*data)
        debug(f"Null loglikelihood = {self.results.null_loglikelihood}")

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return pprint(self.cost)
