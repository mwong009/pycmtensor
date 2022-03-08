# models.py

import traceback

import aesara
import aesara.tensor as aet

from pycmtensor import logger as log

from .configparser import config
from .expressions import Beta, Weights
from .functions import logit, neg_loglikelihood


class PyCMTensorModel:
    def __init__(self, db):
        self.name = "PyCMTensorModel"
        self.params = []  # keep track of params
        self.beta_params = []
        self.inputs = db.get_tensors()
        self.config = config

    def append_to_params(self, params):
        """[Depreciated] Use add_params() instead."""
        log.warning(f"Depreciated method append_to_params(). Use add_params() instead.")
        return self.add_params(params)

    def add_params(self, params):
        if not isinstance(params, (dict, list)):
            msg = "params must be of Type dict or list"
            log.error(msg)
            raise TypeError(msg)

        if isinstance(params, list):
            params = {str(p): p for p in params}

        # update model params into list
        params = self.check_duplicate_param_names(params)
        for param in params:
            if isinstance(param, (Beta, Weights)):
                self.params.append(param)
                if isinstance(param, (Beta)):
                    self.beta_params.append(param)

        if hasattr(self, "cost"):
            self.remove_unused_params(self.cost)

    def check_duplicate_param_names(self, params):
        x = [p for _, p in params.items() if isinstance(p, (Beta, Weights))]
        # check for duplicate params
        seen = set()
        dup = {p.name for p in x if p.name in seen or (seen.add(p.name) or False)}
        if len(dup) > 0:
            msg = f"duplicate param names defined in model: {dup}."
            log.error(msg)
            raise NameError(msg)
        return x

    def remove_unused_params(self, expression):
        """Removes unused parameters not present in `expression`

        Args:
            expression (TensorVariable): The tensor expression to be checked
        """
        stdout = str(aesara.pprint(expression))
        stdout = str.replace(stdout, "(", " ")
        stdout = str.replace(stdout, ")", " ")
        symbols = [s for s in str.split(stdout, " ") if len(s) > 1]
        params = []
        beta_params = []
        unused_params = []
        for param in self.params:
            if param.status == 0:
                if param.name in symbols:
                    params.append(param)
                else:
                    unused_params.append(param.name)
        if len(unused_params) > 0:
            msg = (
                f"Unused Betas from computational graph:"
                + f"".join(f" {p}" for p in unused_params)
                + f" removed. To explicity keep params in model, set param status=1."
            )
            log.warning(msg)

        for param in self.beta_params:
            if param.status == 0:
                if param.name in symbols:
                    beta_params.append(param)

        self.params = params
        self.beta_params = beta_params

    def add_regularizers(self, l_reg):
        if hasattr(self, "cost"):
            self.cost += l_reg
        else:
            log.error("No valid cost function defined.")

    def get_weights(self):
        return [p for p in self.params if ((p().ndim > 1) and (p.status != 1))]

    def get_weight_values(self):
        return [p() for p in self.params if (p().ndim > 1)]

    def get_weight_size(self):
        if len(self.get_weights()) > 0:
            fn = aesara.function(
                [], sum([aet.prod(w.shape) for w in self.get_weights()])
            )
            return fn()
        else:
            return 0

    def get_betas(self):
        return [p for p in self.beta_params if (p.status != 1)]

    def get_beta_values(self):
        return [p() for p in self.beta_params]

    def __repr__(self):
        return f"{self.name}"


class MNLogit(PyCMTensorModel):
    def __init__(self, u, av, database, name="MNLogit"):
        super().__init__(database)
        self.name = name
        # Definition of the choice output (y)
        self.y = database.choiceVar

        # symbolic expression for the choice model probability
        self.p_y_given_x = logit(u, av)

        # symbolic expression for the cost function
        self.cost = neg_loglikelihood(self.p_y_given_x, self.y)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.pred = aet.argmax(self.p_y_given_x, axis=0)

    def prob(self, choice: int = None):
        """Returns the probabilities $P(y_i|x)$ of the slice $i$

        Args:
            choice (int): The index of p_y_given_x. If None, returns the full vector
            of probabilities.
        """
        if choice == None:
            return self.p_y_given_x
        else:
            return self.p_y_given_x[choice]

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return aesara.pprint(self.cost)
