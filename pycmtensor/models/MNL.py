from time import perf_counter
from typing import Dict, List, Tuple, Union

import aesara
import aesara.tensor as aet
from aesara import pprint
from aesara.tensor.var import TensorVariable

from pycmtensor.expressions import Beta, ExpressionParser
from pycmtensor.functions import (
    errors,
    first_order_derivative,
    log_likelihood,
    logit,
    second_order_derivative,
)
from pycmtensor.logger import debug, info
from pycmtensor.models.basic import BaseModel, drop_unused_variables, extract_params
from pycmtensor.utils import time_format


# class MNL(PyCMTensorModel):
class MNL(BaseModel):
    def __init__(
        self,
        ds,
        params: Dict,
        utility: Union[list[TensorVariable], TensorVariable],
        av: List[TensorVariable] = None,
        **kwargs: Tuple,
    ):
        """Defines a Multinomial Logit model

        Args:
            ds (pycmtensor.Data): the database object
            params: dictionary of name:parameter pair
            utility: the vector of utility functions
            av: list of availability conditions. If `None`, all
                availability is set to 1
            **kwargs: Optional keyword arguments for modifying the model configuration settings. See [configuration](../../../user_guide/configuration) in the user guide for details on possible options
        """

        super().__init__(**kwargs)
        self.name = "MNL"
        self.params = []  # keep track of all the Params
        self.betas = []  # keep track of the Betas
        self.updates = []  # keep track of the updates
        self.learning_rate = aet.scalar("learning_rate")

        self.y = ds.y  # Definition of the symbolic choice output (tensor)

        self.p_y_given_x = logit(utility, av)  # expression for the choice probability

        self.ll = log_likelihood(
            self.p_y_given_x, self.y
        )  # expression for the likelihood

        self.cost = -(self.ll / self.y.shape[0])  # expression for the cost

        self.pred = aet.argmax(
            self.p_y_given_x, axis=0
        )  # expression for the prediction

        self.params = extract_params(self.cost, params)
        self.betas = [p for p in self.params if isinstance(p, Beta)]

        # drop unused variables form dataset
        drop_unused = drop_unused_variables(self.cost, self.params, ds())
        ds.drop(drop_unused)

        self.x = ds.x
        info(f"inputs in {self.name}: {self.x}")

        start_time = perf_counter()
        self.build_fns()
        build_time = round(perf_counter() - start_time, 3)

        self.results.build_time = time_format(build_time)
        info(f"Build time = {self.results.build_time}")

    def build_cost_updates_fn(self, updates):
        """Method to call to build/rebuilt cost function with updates to the model. Creates a class function `MNL.cost_updates_fn(*inputs, output, lr)` that receives a list of input variable arrays, the output array, and a learning rate.

        Args:
            updates (List[Tuple[TensorSharedVariable, TensorVariable]]): The list of tuples containing the target shared variable and the new value of the variable.
        """
        self.cost_updates_fn = aesara.function(
            name="cost_updates",
            inputs=self.x + [self.y] + [self.learning_rate],
            outputs=self.cost,
            updates=updates,
        )

    def build_fns(self):
        """Method to call to build mathematical operations without updates to the model. Creates class functions: `MNL.log_likelihood_fn(*inputs, output)`, `MNL.choice_probabilities_fn(*inputs)`, `MNL.choice_predictions_fn(*inputs, output)`, `MNL.prediction_error_fn(*inputs, output)`, `MNL.hessian_fn(*inputs, output)`, `MNL.gradient_vector_fn(*inputs, output)`."""
        self.log_likelihood_fn = aesara.function(
            name="log_likelihood", inputs=self.x + [self.y], outputs=self.ll
        )

        self.choice_probabilities_fn = aesara.function(
            name="choice_probabilities",
            inputs=self.x,
            outputs=self.p_y_given_x.swapaxes(0, 1),
        )

        self.choice_predictions_fn = aesara.function(
            name="choice_predictions", inputs=self.x, outputs=self.pred
        )

        self.prediction_error_fn = aesara.function(
            name="prediction_error",
            inputs=self.x + [self.y],
            outputs=errors(self.p_y_given_x, self.y),
        )

        self.hessian_fn = aesara.function(
            name="hessian",
            inputs=self.x + [self.y],
            outputs=second_order_derivative(-self.cost, self.betas),
            allow_input_downcast=True,
        )

        self.gradient_vector_fn = aesara.function(
            name="gradient_vector",
            inputs=self.x + [self.y],
            outputs=first_order_derivative(-self.cost, self.betas),
            allow_input_downcast=True,
        )

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return pprint(self.cost)
