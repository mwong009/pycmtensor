from time import perf_counter

import aesara
import aesara.tensor as aet
from aesara import pprint

from pycmtensor.expressions import Beta
from pycmtensor.functions import (
    errors,
    first_order_derivative,
    log_likelihood,
    logit,
    second_order_derivative,
)
from pycmtensor.logger import info
from pycmtensor.models.basic import BaseModel, drop_unused_variables, extract_params
from pycmtensor.utils import time_format


class MNL(BaseModel):
    def __init__(self, ds, params, utility, av=None, **kwargs):
        """Defines a Multinomial Logit model

        Args:
            ds (pycmtensor.Data): the database object
            params (Dict): dictionary of name:parameter pair
            utility (Union[list[TensorVariable], TensorVariable]): the vector of utility functions
            av (List[TensorVariable]): list of availability conditions. If `None`, all
                availability is set to 1
            **kwargs (dict): Optional keyword arguments for modifying the model configuration settings. See [configuration](../../../user_guide/configuration) in the user guide for details on possible options

        Attributes:
            x (List[TensorVariable]):
            y (TensorVariable):
            xy (List[TensorVariable]):
            betas (List[Beta]):
            cost (TensorVariable):
            p_y_given_x (TensorVariable):
            ll (TensorVariable):
            pred (TensorVariable):
        """

        super().__init__(**kwargs)
        self.name = "MNL"
        self.params = []  # keep track of all the Params
        self.betas = []  # keep track of the Betas
        self.updates = []  # keep track of the updates
        self.index = aet.ivector("index")  # indices of the dataset
        self.learning_rate = aet.scalar("learning_rate")  # learning rate tensor

        self.y = ds.y  # Definition of the symbolic choice output (tensor)

        self.p_y_given_x = logit(utility, av)  # expression for the choice probability

        # expression for the likelihood
        self.ll = log_likelihood(self.p_y_given_x, self.y, self.index)

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
        self.xy = self.x + [self.y]
        info(f"inputs in {self.name}: {self.x}")

        start_time = perf_counter()
        self.build_fn()
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
            inputs=self.x + [self.y, self.learning_rate, self.index],
            outputs=self.cost,
            updates=updates,
        )

    def build_fn(self):
        """Method to call to build mathematical operations without updates to the model"""

        self.log_likelihood_fn = aesara.function(
            name="log_likelihood", inputs=self.x + [self.y, self.index], outputs=self.ll
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
            inputs=self.x + [self.y, self.index],
            outputs=second_order_derivative(-self.cost, self.betas),
            allow_input_downcast=True,
        )

        self.gradient_vector_fn = aesara.function(
            name="gradient_vector",
            inputs=self.x + [self.y, self.index],
            outputs=first_order_derivative(-self.cost, self.betas),
            allow_input_downcast=True,
        )

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return pprint(self.cost)
