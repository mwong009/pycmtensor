# models.py
import timeit

import aesara.tensor as aet
import dill as pickle
from aesara import function, pprint

from pycmtensor import config
from pycmtensor import logger as log

from .expressions import Beta, Weights
from .functions import (
    bhhh,
    errors,
    full_loglikelihood,
    gradient_norm,
    hessians,
    logit,
    neg_loglikelihood,
)


class PyCMTensorModel:
    def __init__(self, db):
        self.name = "PyCMTensorModel"
        self.config = config
        self.params = []  # keep track of params
        self.beta_params = []
        self.inputs = db.get_tensors()

    def parse_expression(self, expression):
        """Returns a list of symbols from the expression

        Args:
            expression (str): the name of the expression

        Returns:
            list: a list of str words found in expression.

        Note:
            the return list is used to check against unused params, see :meth:`remove_unused_params`
        """
        if hasattr(self, expression):
            _expression = getattr(self, expression)
        stdout = str(pprint(_expression))
        for s in [
            "(",
            ")",
            ",",
            "[",
            "]",
            "{",
            "}",
            "=",
            "Shape",
            "AdvancedSubtensor",
            "Reshape",
            "join",
            "sum",
            "dtype",
            "ARange",
            ":",
            "int64",
            "axis",
            "None",
        ]:
            stdout = str.replace(stdout, s, " ")
        symbols = [s for s in str.split(stdout, " ") if len(s) > 1]
        symbols = list(set(symbols))
        return symbols

    def add_params(self, params):
        """Method to load local variables defined in the main program

        Args:
            params (dict or list): a dict or list of items that is generated from calling :class:`expressions.Beta` or :class:`expressions.Weights`
        """
        if not isinstance(params, (dict, list)):
            msg = "params must be of Type dict or list"
            log.error(msg)
            raise TypeError(msg)

        # create a dict of str(param): param, if params is given as a list
        if isinstance(params, list):
            params = {str(p): p for p in params}

        # remove duplicates
        params = self.check_duplicate_param_names(params)

        # keep track of params
        for _, param in params.items():
            if isinstance(param, (Beta, Weights)):
                self.params.append(param)
                if isinstance(param, (Beta)):
                    self.beta_params.append(param)

        # remove unused Beta params
        self.remove_unused_params(expression="cost")

    def check_duplicate_param_names(self, params):
        x = [p for n, p in params.items() if isinstance(p, (Beta, Weights))]
        # check for duplicate params, raise error and abort if duplicate found
        seen = set()
        dup = {p.name for p in x if p.name in seen or (seen.add(p.name) or False)}
        if len(dup) > 0:
            msg = f"duplicate param names defined in model: {dup}."
            log.error(msg)
            raise NameError(msg)
        return params

    def remove_unused_params(self, expression):
        """Removes unused parameters not present in `expression`

        Args:
            expression (TensorVariable): The tensor expression to be checked
        """
        symbols = self.parse_expression(expression)
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
                f"Unused Betas removed from computational graph: ["
                + f" ,".join(f"{p}" for p in unused_params)
                + f"]. To keep Betas in model, set Beta.status=1"
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
            fn = function([], sum([aet.prod(w.shape) for w in self.get_weights()]))
            return fn()
        else:
            return 0

    def get_betas(self):
        return [p for p in self.beta_params if (p.status != 1)]

    def get_beta_values(self):
        return [p() for p in self.beta_params]

    def build_functions(self, db, optimizer=None):
        """Build callable objects that will calculate ``outputs`` from ``inputs``.

        Args:
            self (PyCMTensorModel): must be of a :class:`PyCMTensorModel` model
                class object.
            db (Database): the :class:`Database` object.
            optimizer (Optimizer, optional): optimizer object class to use. If ``None``
                is given, skips the build of loglikelihood_estimation function.

        Returns:
            self: the updated ``model`` instance.

        Note:
            :func:`build_functions` is called internally. Generally,
            you do not need to call this function in your program.
        """
        log.info("Building model...")
        start_time = timeit.default_timer()
        if optimizer is not None:
            lr = aet.scalar("learning_rate")
            index = aet.lscalar("index")
            batch_size = aet.lscalar("batch_size")
            shift = aet.lscalar("shift")
            db.compile_data()
            opt = optimizer(self.params)
            updates = opt.update(self.cost, self.params, lr)

            self.loglikelihood_estimation = function(
                inputs=[index, batch_size, shift, lr],
                outputs=self.cost,
                updates=updates,
                on_unused_input="ignore",
                givens={
                    t: data[
                        index * batch_size + shift : (index + 1) * batch_size + shift
                    ]
                    for t, data in zip(self.inputs, db.input_shared_data())
                },
            )

        self.loglikelihood = function(
            inputs=[],
            outputs=full_loglikelihood(self.p_y_given_x, self.y),
            on_unused_input="ignore",
            givens={t: data for t, data in zip(self.inputs, db.input_shared_data())},
            name="loglikelihood",
        )
        self.loglikelihood.trust_input = True

        self.output_probabilities = function(
            inputs=self.inputs,
            outputs=self.prob().T,  # returns (N x J) matrix
            on_unused_input="ignore",
            name="p_y_given_x",
        )
        self.output_probabilities.trust_input = True

        self.output_predictions = function(
            inputs=self.inputs,
            outputs=self.pred.T,  # returns (N x 1) matrix
            on_unused_input="ignore",
            name="output_predictions",
        )
        self.output_predictions.trust_input = True

        self.output_estimated_betas = function(
            inputs=[],
            outputs=self.get_beta_values(),
            on_unused_input="ignore",
            name="output_betas",
        )
        self.output_estimated_betas.trust_input = True

        self.output_estimated_weights = function(
            inputs=[],
            outputs=self.get_weight_values(),
            on_unused_input="ignore",
            name="output_weights",
        )
        self.output_estimated_weights.trust_input = True

        self.output_errors = function(
            inputs=[],
            outputs=errors(self.p_y_given_x, self.y),
            on_unused_input="ignore",
            givens={t: data for t, data in zip(self.inputs, db.input_shared_data())},
            name="output_errors",
        )
        self.output_errors.trust_input = True

        self.H = function(
            inputs=[],
            outputs=hessians(self.p_y_given_x, self.y, self.beta_params),
            on_unused_input="ignore",
            givens={t: data for t, data in zip(self.inputs, db.input_shared_data())},
        )
        self.H.trust_input = True

        self.BHHH = function(
            inputs=[],
            outputs=bhhh(self.p_y_given_x, self.y, self.beta_params),
            on_unused_input="ignore",
            givens={t: data for t, data in zip(self.inputs, db.input_shared_data())},
        )
        self.BHHH.trust_input = True

        self.gnorm = function(
            inputs=[],
            outputs=gradient_norm(self.p_y_given_x, self.y, self.beta_params),
            on_unused_input="ignore",
            givens={t: data for t, data in zip(self.inputs, db.input_shared_data())},
        )
        self.gnorm.trust_input = True

        end_time = timeit.default_timer()
        self.build_time = round(end_time - start_time, 3)

    def export_to_pickle(self, f):
        model = self
        model.gnorm = model.gnorm()
        model.H = model.H()
        model.BHHH = model.BHHH()
        del model.loglikelihood_estimation
        del model.loglikelihood
        del model.output_errors

        pickle.dump(model, f)

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
        return pprint(self.cost)


class ResLogitLayer:
    def __init__(self, input, w_in, w_out, activation_in=None, activation_out=None):
        """Definition of the ResLogit neural net layer

        Args:
            input (list or TensorVariable): a list of tensors corresponding to the
            vector of utilities, or a `TensorVariable` vector value.
            w_in (Weights): the :class:`Weights` object for the input side
            w_out (Weights): the :class:`Weights` object for the output side
            activation_in (function, optional): the activation function to use. If
            `None`, use `aet.sigmoid()`
            activation_out (function, optional): the activation function to use. If
            None, use `aet.sigmoid()`

        Attributes:
            output: the output of this layer. Pass this value onto the next layer or
            into the final model.

        Example:
            U = ...
            W_1 = Weights("W_1", (3, 10), 0)
            W_2 = Weights("W_2", (10, 3), 0)
            U = ResLogitLayer(U, W_1, W_2).output
            mymodel = MNLogit(u=U, av=AV, database=db)
        """
        assert w_in.shape[1].eval() == w_out.shape[0].eval()
        if isinstance(input, (list, tuple)):
            assert (
                len(input) == w_in.shape[0].eval()
            ), f"index.0 of w_in must be of the same length as input"
            input = aet.stacklists(input).flatten(2)

        assert isinstance(w_in, (Weights)), "w_in must be Weights()"
        assert isinstance(w_out, (Weights)), "w_out must be  Weights()"
        self.w_in = w_in()
        self.w_out = w_out()
        if activation_in == None:
            activation_in = aet.sigmoid
        if activation_out == None:
            activation_out = aet.sigmoid

        h = activation_in(aet.dot(input.T, self.w_in))
        self.layer_output = activation_out(aet.dot(h, self.w_out)).T
        self.input = input
        self.weights = [self.w_in, self.w_out]
        self.output = self.layer_output + self.input

    def get_layer_outputs(self):
        """Returns the layer output vector. Size of vector is equals to the size of the
        input

        Returns:
            TensorVariable: this layer output
        """
        return self.layer_output
