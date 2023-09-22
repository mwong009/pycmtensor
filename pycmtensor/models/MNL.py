from time import perf_counter

import aesara
import aesara.tensor as aet

from pycmtensor.expressions import Beta
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


class MNL(BaseModel):
    def __init__(self, ds, variables, utility, av=None, **kwargs):
        """Defines a Multinomial Logit model

        Args:
            ds (pycmtensor.Data): the database object
            variables (dict): dictionary containing Param objects or a dictionary of python variables
            utility (Union[list[TensorVariable], TensorVariable]): the vector of utility functions
            av (List[TensorVariable]): list of availability conditions. If `None`, all
                availability is set to 1
            **kwargs (dict): Optional keyword arguments for modifying the model configuration settings. See [configuration](../../../user_guide/configuration.md) in the user guide for details on possible options

        Attributes:
            x (List[TensorVariable]): symbolic variable objects for independent
                variables
            y (TensorVariable): symbolic variable object for the choice variable
            xy (List[TensorVariable]): concatenated list of x and y
            betas (List[Beta]): model beta variables
            cost (TensorVariable): symbolic cost tensor variable function
            p_y_given_x (TensorVariable): probability tensor variable function
            ll (TensorVariable): loglikelihood tensor variable function
            pred (TensorVariable): prediction tensor variable function
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

        self.cost = -self.ll  # expression for the cost

        self.pred = aet.argmax(
            self.p_y_given_x, axis=0
        )  # expression for the prediction

        self.params = self.extract_params(self.cost, variables)
        self.betas = [p for p in self.params if isinstance(p, Beta)]

        # drop unused variables from dataset
        drop_unused = self.drop_unused_variables(self.cost, self.params, ds())
        ds.drop(drop_unused)

        self.x = ds.x
        self.xy = self.x + [self.y]
        info(f"choice: {self.y}")
        info(f"inputs in {self.name}: {self.x}")

        start_time = perf_counter()
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
        """Return the number of estimated betas"""
        return super().n_betas

    def get_betas(self):
        """returns the values of the betas

        Returns:
            (dict): beta values
        """
        return super().get_betas()

    def reset_values(self):
        """resets the values of all parameters"""
        return super().reset_values()

    def build_cost_fn(self):
        """constructs aesara functions for cost and prediction errors"""
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
            outputs=second_order_derivative(self.ll, self.betas),
            allow_input_downcast=True,
        )

        self.gradient_vector_fn = aesara.function(
            name="gradient_vector",
            inputs=self.x + [self.y, self.index],
            outputs=first_order_derivative(self.ll, self.betas),
            allow_input_downcast=True,
        )

    def build_cost_updates_fn(self, updates):
        """build/rebuilt cost function with updates to the model. Creates a class function `MNL.cost_updates_fn(*inputs, output, lr)` that receives a list of input variable arrays, the output array, and a learning rate.

        Args:
            updates (List[Tuple[TensorSharedVariable, TensorVariable]]): The list of tuples containing the target shared variable and the new value of the variable.
        """
        BaseModel.build_cost_updates_fn(self, updates)

    def predict(self, ds, return_probabilities=False):
        """predicts the output of the most likely alternative given the validation dataset in `ds`. The formula is:

        $$
            argmax(p_n(y|x))
        $$

        Args:
            ds (Dataset): pycmtensor dataset
            return_probabilities (bool): if true, returns the probability vector instead

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
            {'pred_choice': array([...])}
            ```

            To return probabilities:
            ```python
            prob = self.predict(ds, return_probabilities=True)
            print(prob)
            ```

            output:
            ```console
            {   0: array([...]),
                1: array([...]),
                ...
            }
            ```
        """
        return BaseModel.predict(self, ds, return_probabilities)

    def elasticities(self, ds, wrt_choice):
        """disaggregated point/cross elasticities of choice y wrt x

        Args:
            ds (pycmtensor.Dataset): dataset containing the training data
            wrt_choice (int): alternative to evaluate the variables on

        Returns:
            (dict): the disaggregate point elasticities of x

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
