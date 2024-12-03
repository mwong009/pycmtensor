from time import perf_counter

import pytensor
import pytensor.tensor as aet

import pycmtensor.models.layers as layers
from pycmtensor.expressions import Beta, Bias, Weight
from pycmtensor.functions import (
    first_order_derivative,
    log_likelihood,
    logit,
    second_order_derivative,
)
from pycmtensor.logger import info
from pycmtensor.models.basic import BaseModel
from pycmtensor.utils import time_format


class ResLogit(BaseModel):
    def __init__(self, ds, variables, utility, av=None, **kwargs):
        BaseModel.__init__(self, ds, variables, utility, av, **kwargs)
        start_time = perf_counter()

        self.name = "ResLogit"
        self.y = ds.y
        self.p_y_given_x = logit(utility, av)
        self.ll = log_likelihood(self.p_y_given_x, self.y, self.index)
        self.cost = -self.ll
        self.pred = aet.argmax(self.p_y_given_x, axis=0)

        self.layer_params = self.extract_layer_params(variables)
        self.params = self.extract_params(self.cost, variables)
        self.params += self.layer_params
        self.betas = [p for p in self.params if isinstance(p, Beta)]
        self.weights = [p for p in self.params if isinstance(p, Weight)]
        self.biases = [p for p in self.params if isinstance(p, Bias)]

        # drop unused variables from dataset
        drop_unused = self.drop_unused_variables(self.cost, self.params, ds())
        ds.drop(drop_unused)

        self.x = ds.x
        self.xy = self.x + [self.y]

        self.build_cost_fn()
        build_time = round(perf_counter() - start_time, 3)
        self.results.build_time = time_format(build_time)
        info(f"inputs in {self.name}: {self.x}")
        info(f"Build time = {self.results.build_time}")

    @property
    def n_params(self):
        """
        Get the total number of estimated parameters in the model.

        Returns:
            int: The total number of estimated parameters.
        """
        return super().n_params

    @property
    def n_betas(self):
        """
        Get the number of estimated Betas in the model.

        Returns:
            int: The number of estimated Betas.
        """
        return super().n_betas

    @property
    def n_weights(self):
        """
        Get the total number of estimated Weight parameters in the model.

        Returns:
            int: The total number of estimated Weight parameters.
        """
        return super().n_weights

    @property
    def n_biases(self):
        """
        Get the total number of estimated Bias parameters in the model.

        Returns:
            int: The total number of estimated Bias parameters.
        """
        return super().n_biases

    def get_betas(self):
        """
        Get a dictionary of Beta values.

        Returns:
            dict: A dictionary of Beta values.
        """
        return super().get_betas()

    def get_weights(self):
        """
        Get a dictionary of Weight values.

        Returns:
            dict: A dictionary of Weight values.
        """
        return super().get_weights()

    def get_biases(self):
        """
        Get a dictionary of Bias values.

        Returns:
            dict: A dictionary of Bias values.
        """
        return super().get_biases()

    def reset_values(self):
        """
        Reset all model parameters to their initial values.
        """
        return super().reset_values()

    def build_cost_fn(self):
        """Constructs pytensor functions for calculating the cost and prediction errors of the ResLogit model.

        Example Usage:
        ```python
        # Create an instance of the ResLogit model
        model = ResLogit(ds, variables, utility, av=None)

        # Call the build_cost_fn method
        model.build_cost_fn()
        ```
        """
        BaseModel.build_cost_fn(self)

    def build_gh_fn(self):
        """Constructs pytensor functions for computing the Hessian matrix and the gradient vector.

        Returns:
            hessian_fn (pytensor function): A function that computes the Hessian matrix.
            gradient_vector_fn (pytensor function): A function that computes the gradient vector.

        !!! note

            The hessians and gradient vector are evaluation at the maximum **log likelihood** estimates instead of the negative loglikelihood, therefore the cost is multiplied by negative one.
        """
        self.hessian_fn = pytensor.function(
            name="hessian",
            inputs=self.x + [self.y, self.index],
            outputs=second_order_derivative(self.ll, self.betas),
            allow_input_downcast=True,
        )

        self.gradient_vector_fn = pytensor.function(
            name="gradient_vector",
            inputs=self.x + [self.y, self.index],
            outputs=first_order_derivative(self.ll, self.betas),
            allow_input_downcast=True,
        )

    def build_cost_updates_fn(self, updates):
        """Build or rebuild the cost function with updates to the model.

        This method creates a class function `ResLogit.cost_updates_fn(*inputs, output, lr)` that takes a list of input variable arrays, an output array, and a learning rate as arguments.

        Args:
            updates (List[Tuple[TensorSharedVariable, TensorVariable]]): A list of tuples containing the target shared variable and the new value of the variable.
        """
        BaseModel.build_cost_updates_fn(self, updates)

    def predict(self, ds):
        """Predicts the output of the most likely alternative given the validation dataset.

        Args:
            ds (Dataset): A pycmtensor dataset object containing the validation data.

        Returns:
            numpy.ndarray: The predicted choices or the vector of probabilities.

        !!! example

            ```python
            # Create an instance of the ResLogit model
            model = ResLogit(ds, variables, utility, av=None)

            # Predict the choices using the predict method
            predictions = self.predict(ds)
            print(predictions)
            ```

            output:
            ```console
            {'pred_choice': array([...])}
            ```

            ```python
            # Predict the probabilities using the predict method
            prob = self.predict(ds)
            print(prob)
            ```

            output:
            ```console
            {   0: array([...]),
                1: array([...]),
                ...
            }
            ```

            The expected output for `predictions` is a dictionary with the key `'pred_choice'` and an array of predicted choices as the value. The expected output for `probabilities` is a dictionary with the keys representing the alternative indices and the values being arrays of probabilities.
        """
        return BaseModel.predict(self, ds)

    def elasticities(self, ds, wrt_choice):
        """Calculate the disaggregated point/cross elasticities of the choice variable `y` with respect to the independent variables `x` in a ResLogit model.

        Args:
            ds (pycmtensor.Dataset): Dataset containing the training data.
            wrt_choice (int): Alternative to evaluate the variables on.

        Returns:
            dict: Disaggregated point elasticities of the independent variables `x`.

        !!! example

            To calculate the elasticity of the choosing alternative 1 w.r.t. (represented by `wrt_choice`) w.r.t. to variable x.

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
    def extract_layer_params(params):
        """Extracts the parameters from the layers of the model"""
        layer_params = []
        if isinstance(params, dict):
            params = [v for _, v in params.items()]
        for p in params:
            if isinstance(p, layers.DenseLayer):
                layer_params.extend(p.params)
        return layer_params
