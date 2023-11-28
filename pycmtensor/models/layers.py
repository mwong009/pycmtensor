# layers.py
"""Model layers"""

import aesara.tensor as aet
import numpy as np

import pycmtensor.functions as functions
from pycmtensor.expressions import Beta, Bias, TensorExpressions, Weight

init_types = ["he", "zeros", "glorot"]


class Layer(TensorExpressions):
    def __init__(self, name, input, status=0, **kwargs):
        """Default Layer class object

        Args:
            name (str): name of the layer for identification
            input (TensorVariable): symbolic variable of the layer input

        Attributes:
            params (list): list of parameters of the layers
            output (TensorVariable): symbolic variable of the layer output
            status (int): status of the Layer, defaults to 0

        """
        self._name = name
        self._input = input
        self._status = status

        if isinstance(input, Layer):
            self._input = input.output

        self.lb = -np.infty
        self.ub = np.infty

        for key, value in kwargs.items():
            if key == "lb":
                self.lb = value
            if key == "ub":
                self.ub = value

    def __repr__(self):
        return f"Layer({self.name}, size={self.n_in, self.n_out})"

    def __call__(self):
        return self._output

    @property
    def name(self):
        return self._name

    @property
    def input(self):
        return self._input

    @property
    def params(self):
        return self._params

    @property
    def output(self):
        return self._output

    @property
    def status(self):
        return self._status

    def get_value(self):
        return {p.name: p.get_value() for p in self.params}

    def set_value(self, param: str, value: np.ndarray):
        for p in self.params:
            if (p.name == param) and (p.shape == value.shape):
                p.set_value(value)


class DenseLayer(Layer):
    def __init__(
        self, name, input, n_in, n_out, init_type=None, activation=None, **kwargs
    ):
        """Class object for dense layer. Compute the function $h=f(w^\\top x + bias)$

        Args:
            name (str): name of the layer
            input (TensorVariable): symbolic variable of the layer input
            n_in (int): the number of input nodes
            n_out (int): the number of output nodes
            init_type (str): initialization type, possible options: `"zeros"`, `"he"`,
                `"glorot"`, `None` (defaut)
            activation (str): activation function $f$, possible options: `"tanh"`,
                `"relu"`, `"sigmoid"`, `"softplus"`, `None` (defaut)
            kwargs(dict): additional keyword arguments:
                - lb: lower bound value of the output(s)
                - ub: upper bound value of the output(s)

        Attributes:
            w (pycmtensor.expressions.Weight): weight matrix of the layer (size=(n_in, n_out))
            bias (pycmtensor.expressions.Bias): bias of the layer (size=(n_out,))
        """
        Layer.__init__(self, name, input, **kwargs)
        self.n_in = n_in
        self.n_out = n_out

        if (not init_type in init_types) and (not init_type is None):
            raise KeyError

        w = Weight(f"{name}_W", (n_in, n_out), init_type=init_type)
        b = Bias(f"{name}_bias", (n_out,))

        if activation == "tanh":
            activation = aet.tanh
        elif activation == "relu":
            activation = functions.relu
        elif activation == "softplus":
            activation = aet.softplus
        elif activation == "sigmoid":
            activation = aet.sigmoid

        self.activation = activation
        self._w = w
        self._bias = b
        self._params = [self.w, self.bias]

        linear = (self.w.T @ self.input) + self.bias  # -> (n_out, len(input))
        if n_out == 1:
            linear = aet.flatten(linear)  # (1, len(input)) -> (len(input),)
        if activation is None:
            self._output = aet.as_tensor_variable(linear)
        else:
            self._output = activation(linear)

        self._output = aet.clip(self._output, self.lb, self.ub)

    def __repr__(self):
        return f"DenseLayer({self.name}, {self.output}; {self.params})"

    def __call__(self):
        return Layer.__call__(self)

    @property
    def w(self):
        return self._w

    @property
    def bias(self):
        return self._bias


class TNBetaLayer(DenseLayer, Beta):
    def __init__(self, name, input, init_type=None, activation=None, **kwargs):
        """TNBeta layer is a dense layer with a output dimension of 1

        Args:
            name (str): name of the layer
            input (TensorVariable): symbolic variable of the layer input
            init_type (str): initialization type, possible options: `"zeros"`, `"he"`,
                `"glorot"`, `None` (defaut)
            activation (str): activation function $f$, possible options: `"tanh"`,
                `"relu"`, `"sigmoid"`, `"softplus"`, `None` (defaut)
            kwargs(dict): additional keyword arguments:
                - lb: lower bound value of the output(s)
                - ub: upper bound value of the output(s)
        """
        if not isinstance(input, Layer):
            raise TypeError(f"input  must be a Layer object. input type={type(input)}")
        DenseLayer.__init__(
            self, name, input, input.n_out, 1, init_type, activation, **kwargs
        )

        def __repr__(self):
            return f"TNBetaLayer({self.name}, {self.output}; {self.params})"

        def __call__(self):
            return DenseLayer.__call__(self)
