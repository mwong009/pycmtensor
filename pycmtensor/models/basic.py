from collections import OrderedDict

import aesara.tensor as aet
import numpy as np
from aesara import function, pprint

import pycmtensor.defaultconfig as defaultconfig
from pycmtensor.expressions import ExpressionParser, Param
from pycmtensor.functions import errors, first_order_derivative, second_order_derivative
from pycmtensor.logger import debug, info, warning
from pycmtensor.models.layers import Layer
from pycmtensor.results import Results

config = defaultconfig.config


class BaseModel(object):
    def __init__(self, ds, variables, utility, av, **kwargs):
        """Basic model class object

        Attributes:
            name (str): name of the model
            config (pycmtensor.Config): pycmtensor config object
            rng (numpy.random.Generator): random number generator
            params (list): list of model parameters (`betas` & `weights`)
            updates (list): list of (param, update) tuples
            learning_rate (TensorVariable): learning rate scalar value
            index (TensorVariable): for indexing the dataset
            is_training (TensorVariable): training mode flag
            results (Results): stores the results of the model estimation
        """
        self.name = ""  # name of the model
        self.config = config  # pycmtensor config object
        self.rng = np.random.default_rng(config.seed)  # random number generator
        self.params = []  # list of model parameters (`betas` & `weights`)
        self.updates = []  # list of (param, update) tuples
        self.learning_rate = aet.scalar("learning_rate")  # learning rate scalar value
        self.index = aet.ivector("index")  # for indexing the dataset
        self.is_training = aet.iscalar("is_training")  # training mode flag
        self.results = Results()  # stores the results of the model estimation

        for key, value in kwargs.items():
            self.config.add(key, value)

        debug(f"Building model...")

        # check if # items in utility is equal to the number of choices
        if set(ds.ds[ds.choice]).issubset(set(range(len(utility)))) is False:
            raise ValueError(
                f"Number of choices in utility function does not match the number of choices in the dataset. Expected {set(range(len(utility)))}, but got {set(ds.ds[ds.choice])}."
            )

    @property
    def n_params(self):
        return self.n_betas + self.n_weights + self.n_biases

    @property
    def n_betas(self):
        if "betas" in dir(self):
            return len(self.betas)
        return 0

    @property
    def n_weights(self):
        if "weights" in dir(self):
            return np.sum([np.prod(w.shape) for w in self.weights])
        return 0

    @property
    def n_biases(self):
        if "biases" in dir(self):
            return np.sum([np.prod(b.shape) for b in self.biases])
        return 0

    def get_weights(self):
        """returns the values of the weights

        Returns:
            (dict): weight values
        """
        if "weights" in dir(self):
            return {w.name: w.get_value() for w in self.weights}
        return {}

    def get_biases(self):
        """returns the values of the biases

        Returns:
            (dict): biases values
        """
        if "biases" in dir(self):
            return {b.name: b.get_value() for b in self.biases}
        return {}

    def get_betas(self):
        """returns the values of the betas

        Returns:
            (dict): beta values
        """
        if "betas" in dir(self):
            return {beta.name: beta.get_value() for beta in self.betas}
        return {}

    def reset_values(self):
        """resets the values of all parameters"""
        if "params" in dir(self):
            for p in self.params:
                p.reset_value()

    def include_params_for_convergence(self, *args, **kwargs):
        """dummy method for additional parameter objects for calculating convergence

        Args:
            *args (None): overloaded arguments
            **kwargs (dict): overloaded keyword arguments

        Returns:
            (OrderedDict): a dictonary of addition parameters to include
        """
        return OrderedDict()

    def include_regularization_terms(self, *regularizers):
        """dummy method for additional regularizers into the cost function

        Args:
            *regularizers (pycmtensor.regularizers.Regularizers): regularizers to
                include in the cost function

        Returns:
            (list[TensorVariable]): a list of symbolic variables that specify additional regualrizers to minimize against
        """

        if len(regularizers) > 0:
            for reg in regularizers:
                self.cost += reg

    def build_cost_fn(self):
        """Constructs Aesara functions for calculating the cost and prediction errors.

        Example Usage:
        ```python
        # Create an instance of the MNL model
        model = MNL(ds, variables, utility, av=None)

        # Call the build_cost_fn method
        model.build_cost_fn()
        ```
        """
        self.log_likelihood_fn = function(
            name="log_likelihood",
            inputs=self.x + [self.y, self.index],
            outputs=self.ll,
            allow_input_downcast=True,
        )

        self.null_log_likelihood_fn = function(
            name="null_log_likelihood",
            inputs=self.x + [self.y, self.index],
            outputs=self.ll,
            allow_input_downcast=True,
            givens={
                param.shared_var: np.zeros_like(param.init_value)
                for param in self.params
            },
        )

        self.prediction_error_fn = function(
            name="prediction_error",
            inputs=self.x + [self.y],
            outputs=errors(self.p_y_given_x, self.y),
            allow_input_downcast=True,
        )

    def build_cost_updates_fn(self, updates):
        """Builds a function that calculates the cost updates for a model.

        Args:
            updates (dict): A dictionary of updates for the model.
        """
        inputs = self.x + [self.y, self.learning_rate, self.index, self.is_training]
        outputs = self.cost

        self.cost_updates_fn = function(
            name="cost_updates",
            inputs=inputs,
            outputs=outputs,
            updates=updates,
            allow_input_downcast=True,
        )

    def build_gh_fn(self):
        """Constructs Aesara functions for computing the Hessian matrix and the gradient vector.

        Returns:
            hessian_fn (Aesara function): A function that computes the Hessian matrix.
            gradient_vector_fn (Aesara function): A function that computes the gradient vector.

        !!! note

            The hessians and gradient vector are evaluation at the maximum **log likelihood** estimates instead of the negative loglikelihood, therefore the cost is multiplied by negative one.
        """
        self.hessian_fn = function(
            name="hessian",
            inputs=self.x + [self.y, self.index],
            outputs=second_order_derivative(self.ll, self.all_betas),
            allow_input_downcast=True,
        )

        self.gradient_vector_fn = function(
            name="gradient_vector",
            inputs=self.x + [self.y, self.index],
            outputs=first_order_derivative(self.ll, self.all_betas),
            allow_input_downcast=True,
        )

    def predict(self, dataset) -> dict:
        """
        Generates predictions on a provided dataset.

        Args:
            dataset (Dataset): The dataset for which predictions are to be generated.

        Returns:
            (dict): A dictionary containing the following key-value pairs:
                '[choice_index]' (list): The model's predicted probabilities for each alternative.
                'pred_[choice_label]' (list): The model's predicted choices.
                'true_[choice_label]' (list): The actual choices from the dataset.
        """
        if not "choice_probabilities_fn" in dir(self):
            self.choice_probabilities_fn = function(
                name="choice_probabilities",
                inputs=self.x,
                outputs=self.p_y_given_x,
                allow_input_downcast=True,
            )

        valid_data = dataset.valid_dataset(self.x)
        valid_ground_truth = dataset.valid_dataset(self.y)
        prob = self.choice_probabilities_fn(*valid_data)
        result = {
            **{i: prob[i] for i in range(prob.shape[0])},
            f"pred_{dataset.choice}": np.argmax(prob, axis=0),
            f"true_{dataset.choice}": valid_ground_truth[0],
        }
        return result

    def elasticities(self, dataset, wrt_choice: int):
        """
        Calculates the elasticities of the model for a specific choice, using a provided dataset.

        Args:
            dataset (Dataset): The dataset to be used in the elasticity calculations.
            wrt_choice (int): The index of the choice for which the elasticities are to be calculated. This index should correspond to a valid choice in the dataset.

        Returns:
            (dict): A dictionary where each key-value pair represents an explanatory variable and its corresponding calculated elasticity. The keys are the names of the explanatory variables, and the values are the calculated elasticities for the specified choice across the dataset.


        """
        p_y_given_x = self.p_y_given_x[self.y, ..., self.index]
        for _ in range(p_y_given_x.ndim - 1):
            p_y_given_x = aet.sum(p_y_given_x, axis=1)
        dy_dx = aet.grad(aet.sum(p_y_given_x), self.x, disconnected_inputs="ignore")

        if not "elasticity_fn" in dir(self):
            self.elasticity_fn = function(
                inputs=self.x + [self.y, self.index],
                outputs={x.name: g * x / p_y_given_x for g, x in zip(dy_dx, self.x)},
                on_unused_input="ignore",
                allow_input_downcast=True,
            )
        train_data = dataset.train_dataset(self.x)
        index = np.arange((len(train_data[-1])))
        choice = (np.ones(shape=index.shape) * wrt_choice).astype(int)
        return self.elasticity_fn(*train_data, choice, index)

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return pprint(self.cost)

    def __getattr__(self, name):
        if name in ["hessian_fn", "gradient_vector_fn"]:
            self.build_gh_fn()
            return getattr(self, name)
        else:
            return False

    @staticmethod
    def extract_params(cost, variables):
        """Extracts Param objects from variables

        Args:
            cost (TensorVariable): function to evaluate
            variables (Union[dict, list]): list of variables from the current program
        """
        params = []
        symbols = ExpressionParser.parse(cost)
        seen = set()

        if isinstance(variables, dict):
            variables = [v for _, v in variables.items()]

        for variable in variables:
            if (not isinstance(variable, Param)) or isinstance(variable, Layer):
                continue

            if isinstance(variable, Param) and (variable.name in seen):
                continue

            if variable.name not in symbols:
                # raise a warning if variable is not in any utility function
                warning(f"{variable.name} not in any utility functions")
                continue

            params.append(variable)
            seen.add(variable.name)

        return params

    @staticmethod
    def drop_unused_variables(cost, params, variables):
        """Internal method to remove ununsed tensors

        Args:
            cost (TensorVariable): function to evaluate
            params (Param): param objects
            variables (dict): list of array variables from the dataset

        Returns:
            (list): a list of param names which are not used in the model
        """
        symbols = ExpressionParser.parse(cost)
        param_names = [p.name for p in params]
        symbols = [s for s in symbols if s not in param_names]
        return [var for var in list(variables) if var not in symbols]
