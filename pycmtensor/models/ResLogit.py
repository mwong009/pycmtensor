import aesara.tensor as aet
from aesara.tensor.sharedvar import TensorSharedVariable
from aesara.tensor.var import TensorVariable

from ..expressions import Weight


class ResLogitLayer:
    def __init__(
        self,
        input,
        w_in,
        w_out,
        bias_in=None,
        bias_out=None,
        activation_in=None,
        activation_out=None,
    ):
        """Definition of the ResLogit neural net layer

        Args:
            input (list or TensorVariable): a list of tensors corresponding to the
                vector of utilities, or a ``TensorVariable`` vector
            w_in (Weight): the :class:`Weight` object for the input side
            w_out (Weight): the :class:`Weight` object for the output side
            bias_in (Bias): TODO
            bias_out (Bias): TODO
            activation_in (function, optional): the 1st activation function to use. If
                ``None``, defaults to ``aet.sigmoid()``
            activation_out (function, optional): the 2nd activation function to use. If
                ``None``, defaults to ``aet.sigmoid()``

        Example:
            ``U = ...``
            ``W_1 = Weight("W_1", (3, 10), init_type='he')``
            ``W_2 = Weight("W_2", (10, 3), init_type='he')``
            ``U = ResLogitLayer(U, W_1, W_2).output``
            ``mymodel = MNL(u=U, av=AV, database=db)``
        """
        if not w_in.shape[1].eval() == w_out.shape[0].eval():
            raise ValueError(f"last dim of w_in must match first dim of w_out")

        if isinstance(input, (list, tuple)):
            if not len(input) == w_in.shape[0].eval():
                msg = f"index.0 of w_in must be of the same length as input"
                raise ValueError(msg)
            input = aet.stack(input)

        if not (isinstance(w_in, (Weight)) and isinstance(w_out, (Weight))):
            msg = f"w_in and w_out must be expressions.Weight class object"
            raise ValueError(msg)

        for bias in [bias_in, bias_out]:
            if bias is not None:
                if not isinstance(bias, (TensorSharedVariable)):
                    msg = f"{bias.name} must be a TensorSharedVariable class"
                    raise ValueError(msg)
                if not bias.ndim == 1:
                    msg = f"{bias.name} must be a Vector with ndim=1"
                    raise ValueError(msg)

        if activation_in == None:
            if w_in.init_value == "he":
                activation_in = aet.nnet.relu
            else:
                activation_in = aet.tanh
        if activation_out == None:
            if w_in.init_value == "he":
                activation_out = aet.nnet.relu
            else:
                activation_out = aet.tanh

        self._updates = []
        self._params = []
        self.input = input

        self.w_in = w_in()
        self.w_out = w_out()

        self.h = activation_in(aet.dot(self.w_in.T, self.input))
        self._params.append(self.w_in)
        if bias_in is not None:
            self.h = self.h + aet.expand_dims(bias_in, -1)
            self._params.append(bias_in)

        self.h2 = activation_out(aet.dot(self.w_out.T, self.h))
        self._params.append(self.w_in)
        if bias_out is not None:
            self.h2 = self.h2 + aet.expand_dims(bias_out, -1)
            self._params.append(bias_out)

        self._output = self.h2 + self.input

    @property
    def updates(self) -> list[(TensorVariable, TensorVariable)]:
        """Returns a list of tuples containing the old parameter and the new
        parameter"""
        return self._updates

    @property
    def params(self) -> list[TensorSharedVariable]:
        """Returns the list of ``TensorSharedVariable`` layer parameters in sequence"""
        return self._params

    def output(self, mode=None) -> TensorVariable:
        """Returns the output vector. Size of vector is equals to the size of the
        input"""
        return self._output


"""
reslogit = ResLogit(name=name, utility=utility, av=av, params=params, db=db)

bn_gamma = expressions.Param('bn_gamma', np.ones(3, dtype=FLOATX))
bn_beta = expressions.Param('bn_beta', np.zeros(3, dtype=FLOATX))
reslogit.add_layer(BatchNormLayer(utility, bn_gamma, bn_beta))

w_1 = expressions.Weight(name="w_1", size=(3, 10), init_type="he")
b_1 = expressions.Bias(name="b_1", size=(10,))
reslogit.add_layer(DenseLayer(reslogit.layers[-1], w_1, b_1, activation=relu))

w_2 = expressions.Weight(name="w_2", size=(10, 3), init_type="he")
b_2 = expressions.Bias(name="b_2", size=(3,))
reslogit.add_layer(DenseLayer(reslogit.layers[-1], w_2, b_2, activation=relu))

"""
