from time import perf_counter

import aesara.tensor as aet

from pycmtensor.expressions import Beta
from pycmtensor.functions import log_likelihood, logit
from pycmtensor.logger import info
from pycmtensor.models.basic import BaseModel
from pycmtensor.utils import time_format


class MNL(BaseModel):
    def __init__(self, ds, variables, utility, av=None, **kwargs):
        """Initialize the Multinomial Logit model with the given dataset, variables, utility function, and availability.

        Args:
            ds (pycmtensor.Data): the database object
            variables (dict): dictionary containing Param objects or a dictionary of python variables
            utility (Union[list[TensorVariable], TensorVariable]): the vector of utility functions
            av (List[TensorVariable]): list of availability conditions. If `None`, all
                availability is set to 1
            **kwargs (dict): Optional keyword arguments for modifying the model configuration settings. See [configuration](../../../user_guide/configuration.md) in the user guide for details on possible options

        Attributes:
            name (str): name of the model
            y (TensorVariable): symbolic variable object for the choice variable
            p_y_given_x (TensorVariable): probability tensor variable function
            ll (TensorVariable): loglikelihood tensor variable function
            cost (TensorVariable): symbolic cost tensor variable function
            pred (TensorVariable): prediction tensor variable function
            params (list): list of model parameters (`betas`)
            betas (List[Beta]): model beta variables
            x (List[TensorVariable]): symbolic variable objects for independent
                variables
            xy (List[TensorVariable]): concatenated list of x and y
        """

        BaseModel.__init__(self, ds, variables, utility, av, **kwargs)
        start_time = perf_counter()

        self.name = "MNL"
        self.y = ds.y
        self.p_y_given_x = logit(utility, av)
        self.ll = log_likelihood(self.p_y_given_x, self.y, self.index)
        self.cost = -self.ll
        self.pred = aet.argmax(self.p_y_given_x, axis=0)

        self.params = self.extract_params(self.cost, variables)
        self.betas = [p for p in self.params if isinstance(p, Beta)]

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
    def all_betas(self):
        """Return all the beta parameters in the model.

        Returns:
            (List[Beta]): A list of all beta parameters in the model.
        """
        return self.betas

    @property
    def n_params(self):
        """Returns the number of parameters in the Multinomial Logit model.

        Returns:
            (int): The number of parameters in the Multinomial Logit model.
        """
        return super().n_params

    @property
    def n_betas(self):
        """Return the number of estimated betas in the Multinomial Logit model.

        Returns:
            (int): The number of estimated betas.
        """
        return super().n_betas

    def get_betas(self):
        """Returns the values of the betas in the model as a dictionary.

        Returns:
            (dict): A dictionary containing the beta values, where the keys represent the beta names and the values represent their corresponding values.
        """
        return super().get_betas()

    def reset_values(self) -> None:
        """Resets the values of all parameters by calling the reset_values method of the parent class.

        This method resets the values of all parameters to their initial values.

        Example Usage:
        ```python
        # Create an instance of the MNL class
        model = MNL(ds, variables, utility, av=None)

        # Call the reset_values method to reset the parameter values
        model.reset_values()
        ```

        Inputs:
        - None

        Flow:
        1. The `reset_values` method is called.
        2. The method calls the `reset_values` method of the parent class `BaseModel` to reset the values of all parameters.

        Outputs:
        - None
        """
        return super().reset_values()

    def build_cost_fn(self):
        """Constructs Aesara functions for calculating the cost and prediction errors of the Multinomial Logit model.

        Example Usage:
        ```python
        # Create an instance of the MNL class
        model = MNL(ds, variables, utility, av=None)

        # Call the build_cost_fn method
        model.build_cost_fn()
        ```
        """
        BaseModel.build_cost_fn(self)

    def build_gh_fn(self):
        """Constructs Aesara functions for computing the Hessian matrix and the gradient vector.

        Returns:
            hessian_fn (Aesara function): A function that computes the Hessian matrix.
            gradient_vector_fn (Aesara function): A function that computes the gradient vector.

        !!! note

            The hessians and gradient vector are evaluation at the maximum **log likelihood** estimates instead of the negative loglikelihood, therefore the cost is multiplied by negative one.
        """
        BaseModel.build_gh_fn(self)

    def build_cost_updates_fn(self, updates):
        """Build or rebuild the cost function with updates to the model.

        This method creates a class function `MNL.cost_updates_fn(*inputs, output, lr)` that takes a list of input variable arrays, an output array, and a learning rate as arguments.

        Args:
            updates (List[Tuple[TensorSharedVariable, TensorVariable]]): A list of tuples containing the target shared variable and the new value of the variable.
        """
        BaseModel.build_cost_updates_fn(self, updates)

    def predict(self, ds):
        """Predicts the output of the most likely alternative given the validation dataset.

        Args:
            ds (Dataset): A pycmtensor dataset object containing the validation data.

        Returns:
            (numpy.ndarray): The predicted choices or the vector of probabilities.

        !!! example

            ```python
            # Create an instance of the MNL class
            model = MNL(ds, variables, utility, av=None)

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
        """Calculate the disaggregated point/cross elasticities of the choice variable `y` with respect to the independent variables `x` in a Multinomial Logit model.

        Args:
            ds (pycmtensor.Dataset): Dataset containing the training data.
            wrt_choice (int): Alternative to evaluate the variables on.

        Returns:
            (dict): Disaggregated point elasticities of the independent variables `x`.

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
