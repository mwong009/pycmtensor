# models.py

import aesara
import aesara.tensor as aet

from pycmtensor.expressions import Beta, Weights
from pycmtensor.functions import logit, neg_loglikelihood


class PyCMTensorModel:
    def __init__(self, db):
        self.name = "PyCMTensorModel"
        self.params = []  # keep track of params
        self.beta_params = []
        self.inputs = db.tensors()

    def append_to_params(self, params):
        """[Depreciated] Use add_params() instead."""
        print(f"Depreciated method append_to_params(). Use add_params() instead.")
        return self.add_params(params)

    def add_params(self, params):
        assert isinstance(params, (dict, list)), f"params must be of type dict or list"
        if isinstance(params, list):
            d = {}
            for param in params:
                d[str(param)] = param
            params = d

        for _, param in params.items():
            if isinstance(param, (Beta, Weights)):
                # update model params into list
                self.params.append(param)
                if isinstance(param, (Beta)):
                    self.beta_params.append(param)

    def add_regularizers(self, l_reg):
        if hasattr(self, "cost"):
            self.cost += l_reg
        else:
            print("Error in reading cost function")

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

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return aesara.pprint(self.cost)
