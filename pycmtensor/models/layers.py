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


# class BatchNormLayer(Layer):
#     def __init__(self, gamma, beta, batch_size, factor=0.05, epsilon=1e-6):
#         """Class object for Batch Normalization layer

#         Args:
#             gamma (TensorSharedVariable): gamma variable for variance
#             beta (TensorSharedVariable): beta variable for mean
#             batch_size (int): batch size indicator
#             factor (float, optional): exponential moving average factor
#             epsilon (float, optional): small value to prevent floating point error

#         Notes:
#             The ema factor controls how fast/slow the running average is changed.
#             Higher ``factor`` value discounts older values faster.
#         """

#         self._updates = []
#         self.batch_size = batch_size
#         self.gamma = gamma
#         self.beta = beta
#         self.epsilon = epsilon
#         self.factor = factor
#         self.gamma = gamma
#         self.beta = beta
#         self.params = [self.gamma, self.beta]

#         # internal record of the running variance and mean
#         self._mv_var = aesara.shared(np.ones(gamma.shape), name="mv_var")
#         self._mv_mean = aesara.shared(np.zeros(beta.shape), name="mv_mean")

#     def apply(self, input):
#         """Function to apply the input to the computational graph"""
#         if isinstance(input, (list, tuple)):
#             input = aet.stack(input)
#         self.input = input

#         # variance and mean of each batch of input during training
#         batch_var = aet.var(self.input, axis=1)
#         batch_mean = aet.mean(self.input, axis=1)

#         # updates for the running mean and variance values
#         ema_var = functions.exp_mov_average(batch_var, self._mv_mean, alpha=self.factor)
#         ema_mean = functions.exp_mov_average(
#             batch_mean, self._mv_var, alpha=self.factor
#         )
#         self._updates.append((self._mv_var, ema_mean))
#         self._updates.append((self._mv_mean, ema_var))

#         # condition when training
#         batch_std = aet.shape_padaxis(aet.sqrt(batch_var + self.epsilon), 1)
#         h = (self.input - aet.shape_padaxis(batch_mean, 1)) / batch_std
#         batch_norm = self.gamma() * h.swapaxes(0, -1) + self.beta()
#         self.batch_norm = batch_norm.swapaxes(0, -1)

#         # condition when testing
#         mv_std = aet.shape_padaxis(aet.sqrt(self.mv_var + self.epsilon), 1)
#         h_hat = (self.input - aet.shape_padaxis(self.mv_mean, 1)) / mv_std
#         full_norm = self.gamma() * h_hat.swapaxes(0, -1) + self.beta()
#         self.full_norm = full_norm.swapaxes(0, -1)

#     @property
#     def mv_mean(self):
#         """Returns the stored running mean"""
#         return self._mv_mean

#     @property
#     def mv_var(self):
#         """Return the stored running variance"""
#         return self._mv_var

#     @property
#     def updates(self):
#         """Returns a list of update tuple pairs"""
#         return self._updates

#     @property
#     def output(self):
#         """Returns the output of this layer

#         Note:
#             Returns the full normalized layer using the running mean if the input
#             length is not equivalent to the batch size
#         """
#         return aet.switch(
#             aet.eq(self.input.shape[1], aet.constant(self.batch_size)),
#             self.batch_norm,
#             self.full_norm,
#         )


# class ResidualLayer:
#     def __init__(self, layers: list):
#         """Definition of the Residual layer block

#         Args:
#             layers (list): a list of layers that defines the residual block

#         Example:

#             .. code-block:: python

#                 res_layer = ResidualLayer(layers=[
#                     DenseLayer(w_1, b_1, activation=relu),
#                     DenseLayer(w_2, b_2, activation=relu)
#                 ])
#         """
#         for layer in layers:
#             if not isinstance(layer, Layer):
#                 raise TypeError(f"{layer} is not a Layer class instance")

#         self.layers = layers
#         self.params = []
#         self._updates = []

#     def apply(self, input):
#         """Function to apply the input to the computational graph"""
#         if isinstance(input, (list, tuple)):
#             input = aet.stack(input)
#         self.input = input

#         for n, layer in enumerate(self.layers):
#             if n == 0:
#                 layer.apply(self.input)
#             else:
#                 layer.apply(self.layers[n - 1].output)
#             self.params.extend(layer.params)
#             self._updates.extend(layer.updates)
#         self._output = self.layers[-1].output + self.input

#     @property
#     def updates(self):
#         """Returns a list of update tuple pairs"""
#         return self._updates

#     def output(self):
#         """Returns the output of this layer"""
#         return self._output
