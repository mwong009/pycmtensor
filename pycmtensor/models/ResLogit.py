import aesara.tensor as aet

from ..expressions import Weights


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
            mymodel = MNL(u=U, av=AV, database=db)
        """
        assert w_in.shape[1].eval() == w_out.shape[0].eval()
        if isinstance(input, (list, tuple)):
            assert (
                len(input) == w_in.shape[0].eval()
            ), f"index.0 of w_in must be of the same length as input"
            input = aet.stack(input).flatten(2)

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
