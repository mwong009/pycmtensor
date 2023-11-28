from collections import OrderedDict
from time import perf_counter

import aesara
import aesara.tensor as aet
import numpy as np
from aesara import pprint

import pycmtensor.models.layers as layers
from pycmtensor.expressions import Beta, Bias, Weight
from pycmtensor.functions import (
    errors,
    first_order_derivative,
    log_likelihood,
    logit,
    second_order_derivative,
)
from pycmtensor.logger import info
from pycmtensor.models.basic import BaseModel
from pycmtensor.utils import time_format


class TasteNet(BaseModel):
    def __init__(self, ds, params, utility, av=None, **kwargs):
        super().__init__(**kwargs)
        self.name = "TasteNet"
        self.params = []
        self.weights = []
        self.biases = []
        self.betas = []
        self.updates = []
        self.index = aet.ivector("index")
        self.learning_rate = aet.scalar("learning_rate")

        self.y = ds.y
        self.p_y_given_x = logit(utility, av)
        self.ll = log_likelihood(self.p_y_given_x, self.y, self.index)
        self.cost = -self.ll
        self.pred = aet.argmax(self.p_y_given_x, axis=0)

        start_time = perf_counter()

        self.layer_params = self.extract_layer_params(params)
        self.tn_betas = self.extract_tn_outputs(params)
        self.params = self.extract_params(self.cost, params)
        self.params += self.layer_params
        self.betas = [p for p in self.params if isinstance(p, Beta)]
        self.weights = [p for p in self.params if isinstance(p, Weight)]
        self.biases = [p for p in self.params if isinstance(p, Bias)]

        drop_unused = self.drop_unused_variables(self.cost, self.params, ds())
        ds.drop(drop_unused)

        self.x = ds.x
        self.xy = self.x + [self.y]
        info(f"choice: {self.y}")
        info(f"inputs in {self.name}: {self.x}")

        self.build_cost_fn()

        build_time = round(perf_counter() - start_time, 3)

        self.results.build_time = time_format(build_time)
        info(f"Build time = {self.results.build_time}")

    @property
    def n_params(self):
        """Return the total number of estimated parameters"""
        return super().n_params

    @property
    def n_betas(self):
        """Return the number of estimated Betas"""
        return super().n_betas

    @property
    def n_weights(self):
        """Return the total number of estimated Weight parameters"""
        return super().n_weights

    @property
    def n_biases(self):
        """Return the total number of estimated Weight parameters"""
        return super().n_biases

    def get_betas(self):
        """Returns a dict of Beta values"""
        return super().get_betas()

    def get_weights(self):
        """Returns a dict of Weight values"""
        return super().get_weights()

    def get_biases(self):
        """Returns a dict of Weight values"""
        return super().get_biases()

    def reset_values(self):
        """Resets Model parameters to their initial value"""
        return super().reset_values()

    def include_params_for_convergence(self, data, index):
        """Returns a Ordered dict of parameters values to check for convergence

        Returns:
            (OrderedDict): ordered dictionary of parameter values
        """
        params = OrderedDict()
        for key, value in self.tn_betas_fn(*data, index).items():
            mv = np.array(np.median(value))
            params[key] = mv

        return params

    def build_cost_fn(self):
        """constructs aesara functions for cost and prediction errors"""
        self.tn_betas_fn = aesara.function(
            name="tn_params",
            inputs=self.x + [self.y, self.index],
            outputs={layer.name: layer.output for layer in self.tn_betas},
            allow_input_downcast=True,
        )

        self.log_likelihood_fn = aesara.function(
            name="log_likelihood",
            inputs=self.x + [self.y, self.index],
            outputs=self.ll,
            allow_input_downcast=True,
        )

        self.prediction_error_fn = aesara.function(
            name="prediction_error",
            inputs=self.x + [self.y],
            outputs=errors(self.p_y_given_x, self.y),
            allow_input_downcast=True,
        )

    def build_gh_fn(self):
        """constructs aesara functions for hessians and gradient vectors

        !!! note

            The hessians and gradient vector are evaluation at the maximum **log likelihood** estimates instead of the negative loglikelihood, therefore the cost is multiplied by negative one.
        """
        self.hessian_fn = aesara.function(
            name="hessian",
            inputs=self.x + [self.y, self.index],
            outputs=second_order_derivative(self.ll, self.betas + self.tn_betas),
            allow_input_downcast=True,
        )

        self.gradient_vector_fn = aesara.function(
            name="gradient_vector",
            inputs=self.x + [self.y, self.index],
            outputs=first_order_derivative(self.ll, self.betas + self.tn_betas),
            allow_input_downcast=True,
        )

    def build_cost_updates_fn(self, updates):
        """build/rebuilt cost function with updates to the model. Creates a class function `MNL.cost_updates_fn(*inputs, output, lr)` that receives a list of input variable arrays, the output array, and a learning rate.

        Args:
            updates (List[Tuple[TensorSharedVariable, TensorVariable]]): The list of tuples containing the target shared variable and the new value of the variable.
        """
        BaseModel.build_cost_updates_fn(self, updates)

    def predict(self, ds):
        """predicts the output of the most likely alternative given the validation dataset in `ds`. The formula is:

        $$
            argmax(p_n(y|x))
        $$

        Args:
            ds (Dataset): pycmtensor dataset

        Returns:
            (numpy.ndarray): the predicted choices or the vector of probabilities

        !!! example

            To return predicted choices:
            ```python
            predictions = self.predict(ds)
            print(predictions)
            ```

            output:
            ```console
            array([...])
            ```
        """
        return BaseModel.predict(self, ds)

    def elasticities(self, ds, wrt_choice):
        """disaggregate point/cross elasticities of choice y wrt x

        Args:
            ds (pycmtensor.Dataset): dataset containing the training data
            wrt_choice (int): alternative to evaluate the variables on

        Returns:
            (dict): the disaggregate point elasticities of x

        !!! example

            To calculate the elasticity of the choosing alternative 1 w.r.t. (represented by `wrt_choice`) w.r.t. to variable x.
            :
            ```python
            disag_elas = self.elasticities(ds, wrt_choice=1)
            ```

            output:
            ```console
            {
                'variable_1': array([...]),
                'variable_2': array([...]),
                ...
            }
            ```

            The return values in the dictionary are the disaggregated elasticities. To calculate the aggregated elasticities, use `np.mean()`.

        !!! note

            This function returns the elasticities for *all* variables. To obtain the point or cross elasticities, simply select the appropriate dictionary key from the output (`wrt_choice` w.r.t. x).
        """
        return BaseModel.elasticities(self, ds, wrt_choice)

    @staticmethod
    def extract_tn_outputs(params):
        tn_betas = []
        if isinstance(params, dict):
            params = [v for _, v in params.items()]
        for beta in params:
            if isinstance(beta, layers.TNBetaLayer):
                tn_betas.append(beta)

        return tn_betas

    @staticmethod
    def extract_layer_params(params):
        layer_params = []
        if isinstance(params, dict):
            params = [v for _, v in params.items()]
        for p in params:
            if isinstance(p, layers.DenseLayer):
                layer_params.extend(p.params)
        return layer_params
