from collections import OrderedDict
from time import perf_counter

import numpy as np
import pytensor
import pytensor.tensor as aet

import pycmtensor.models.layers as layers
from pycmtensor.expressions import Beta, Bias, Weight
from pycmtensor.functions import log_likelihood, logit
from pycmtensor.logger import info
from pycmtensor.models.basic import BaseModel
from pycmtensor.utils import time_format


class TasteNet(BaseModel):
    """
    TasteNet is a discrete choice model based on the TasteNetMNL model (Han et al., 2022).

    Reference:
    Han, Y., Pereira, F. C., Ben-Akiva, M., & Zegras, C. (2022). A neural-embedded discrete choice model: Learning taste representation with strengthened interpretability. Transportation Research Part B: Methodological, 163, 166-186.

    """

    def __init__(self, ds, variables, utility, av=None, **kwargs):
        """
        Initialize the TasteNet model with the given dataset, parameters, utility function, and availability.

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
            params (list): list of model parameters (`betas` + `tn_betas`)
            betas (List[Beta]): model beta variables
            weights (List[Weight]): model weight variables
            biases (List[Bias]): model bias variables
            x (List[TensorVariable]): symbolic variable objects for independent
                variables
            xy (List[TensorVariable]): concatenated list of x and y
        """
        BaseModel.__init__(self, ds, variables, utility, av, **kwargs)
        start_time = perf_counter()

        self.name = "TasteNet"
        self.y = ds.y
        self.p_y_given_x = logit(utility, av)
        self.ll = log_likelihood(self.p_y_given_x, self.y, self.index)
        self.cost = -self.ll
        self.pred = aet.argmax(self.p_y_given_x, axis=0)

        self.layer_params = self.extract_layer_params(variables)
        self.tn_betas = self.extract_tn_outputs(variables)
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
    def all_betas(self):
        """Return all the beta parameters in the model.

        Returns:
            List[Beta]: A list of all beta parameters in the model.
        """
        return self.betas + self.tn_betas

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

    def include_params_for_convergence(self, data, index):
        """
        Get an ordered dictionary of parameter values to check for convergence.

        Args:
            data: The data used for the model.
            index: The index of the data.

        Returns:
            OrderedDict: An ordered dictionary of parameter values.
        """
        params = OrderedDict()
        for key, value in self.tn_betas_fn(*data, index).items():
            mv = np.array(np.median(value))
            params[key] = mv

        return params

    def build_cost_fn(self):
        """Constructs pytensor functions for calculating the cost and prediction errors of the TasteNet model.

        Example Usage:
        ```python
        # Create an instance of the TasteNet model
        model = TasteNet(ds, variables, utility, av=None)

        # Call the build_cost_fn method
        model.build_cost_fn()
        ```
        """
        self.tn_betas_fn = pytensor.function(
            name="tn_params",
            inputs=self.x + [self.y, self.index],
            outputs={layer.name: layer.output for layer in self.tn_betas},
            allow_input_downcast=True,
        )

        BaseModel.build_cost_fn(self)

    def build_gh_fn(self):
        """Constructs pytensor functions for computing the Hessian matrix and the gradient vector.

        Returns:
            hessian_fn (pytensor function): A function that computes the Hessian matrix.
            gradient_vector_fn (pytensor function): A function that computes the gradient vector.

        !!! note

            The hessians and gradient vector are evaluation at the maximum **log likelihood** estimates instead of the negative loglikelihood, therefore the cost is multiplied by negative one.
        """
        BaseModel.build_gh_fn(self)

    def build_cost_updates_fn(self, updates):
        """Build or rebuild the cost function with updates to the model.

        This method creates a class function `TasteNet.cost_updates_fn(*inputs, output, lr)` that takes a list of input variable arrays, an output array, and a learning rate as arguments.

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
            # Create an instance of the TasteNet model
            model = TasteNet(ds, variables, utility, av=None)

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
        """Calculate the disaggregated point/cross elasticities of the choice variable `y` with respect to the independent variables `x` in a choice model.

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
    def extract_tn_outputs(params):
        """Extracts the output from the layers of the model"""
        tn_betas = []
        if isinstance(params, dict):
            params = [v for _, v in params.items()]
        for beta in params:
            if isinstance(beta, layers.TNBetaLayer):
                tn_betas.append(beta)

        return tn_betas

    @staticmethod
    def extract_layer_params(params):
        """Extracts the parameters from the layers of the model"""
        layer_params = []
        if isinstance(params, dict):
            params = [v for _, v in params.items()]
        for p in params:
            if isinstance(p, layers.Layer):
                layer_params.extend(p.params)
        return layer_params
