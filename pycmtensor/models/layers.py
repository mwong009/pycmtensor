# layers.py
"""Model layers"""
import aesara
import aesara.tensor as aet
import numpy as np
from aesara.tensor.math import sigmoid, tanh
from aesara.tensor.nnet import relu

from ..functions import exp_mov_average
from ..logger import debug


class Layer:
    """Default class type"""

    pass


class DenseLayer(Layer):
    def __init__(self, w, bias, activation=None):
        """Class object for dense layer

        Args:
            w (TensorSharedVariable): layer weights with ndim=2
            bias (TensorSharedVariable): layer bias with ndim=1
            activation: the activation function, possible options are ``tanh``,
                ``relu``, ``sigm``, ``None``

        Note:
            Layer activation function is set based on the type of weight initialization.
            If weight init is "he", the activation is relu, if "glorot", the activation
            is tanh, otherwise the activation defaults to sigm.
            Setting activation to other than ``None`` overrides this.
        """

        if activation is None:
            if w.init_type == "he":
                activation = relu
                debug(f"activation of DenseLayer({w.shape}) set as ReLU")
            elif w.init_type == "glorot":
                activation = tanh
                debug(f"activation of DenseLayer({w.shape}) set as tanh")
            else:
                activation = sigmoid
                debug(f"activation of DenseLayer({w.shape}) set as sigm")
        else:
            activation = activation

        self.activation = activation
        self.w = w
        self.bias = bias
        self.params = [w, bias]

    def apply(self, input):
        """Function to apply the input to the computational graph"""
        if isinstance(input, (list, tuple)):
            input = aet.stack(input)
        self.input = input

        h = aet.dot(self.input.swapaxes(0, -1), self.w()) + self.bias()
        self._output = self.activation(h).swapaxes(0, -1)

    @property
    def updates(self):
        """Returns a list of update tuple pairs"""
        return [()]

    @property
    def output(self):
        """Returns the output of this layer"""
        return self._output


class BatchNormLayer(Layer):
    def __init__(self, gamma, beta, batch_size, factor=0.05, epsilon=1e-6):
        """Class object for Batch Normalization layer

        Args:
            gamma (TensorSharedVariable): gamma variable for variance
            beta (TensorSharedVariable): beta variable for mean
            batch_size (int): batch size indicator
            factor (float, optional): exponential moving average factor
            epsilon (float, optional): small value to prevent floating point error

        Notes:
            The ema factor controls how fast/slow the running average is changed.
            Higher ``factor`` value discounts older values faster.
        """

        self._updates = []
        self.batch_size = batch_size
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.factor = factor
        self.gamma = gamma
        self.beta = beta
        self.params = [self.gamma, self.beta]

        # internal record of the running variance and mean
        self._mv_var = aesara.shared(np.ones(gamma.shape), name="mv_var")
        self._mv_mean = aesara.shared(np.zeros(beta.shape), name="mv_mean")

    def apply(self, input):
        """Function to apply the input to the computational graph"""
        if isinstance(input, (list, tuple)):
            input = aet.stack(input)
        self.input = input

        # variance and mean of each batch of input during training
        batch_var = aet.var(self.input, axis=1)
        batch_mean = aet.mean(self.input, axis=1)

        # updates for the running mean and variance values
        ema_var = exp_mov_average(batch_var, self._mv_mean, alpha=self.factor)
        ema_mean = exp_mov_average(batch_mean, self._mv_var, alpha=self.factor)
        self._updates.append((self._mv_var, ema_mean))
        self._updates.append((self._mv_mean, ema_var))

        # condition when training
        batch_std = aet.shape_padaxis(aet.sqrt(batch_var + self.epsilon), 1)
        h = (self.input - aet.shape_padaxis(batch_mean, 1)) / batch_std
        batch_norm = self.gamma() * h.swapaxes(0, -1) + self.beta()
        self.batch_norm = batch_norm.swapaxes(0, -1)

        # condition when testing
        mv_std = aet.shape_padaxis(aet.sqrt(self.mv_var + self.epsilon), 1)
        h_hat = (self.input - aet.shape_padaxis(self.mv_mean, 1)) / mv_std
        full_norm = self.gamma() * h_hat.swapaxes(0, -1) + self.beta()
        self.full_norm = full_norm.swapaxes(0, -1)

    @property
    def mv_mean(self):
        """Returns the stored running mean"""
        return self._mv_mean

    @property
    def mv_var(self):
        """Return the stored running variance"""
        return self._mv_var

    @property
    def updates(self):
        """Returns a list of update tuple pairs"""
        return self._updates

    @property
    def output(self):
        """Returns the output of this layer

        Note:
            Returns the full normalized layer using the running mean if the input
            length is not equivalent to the batch size
        """
        return aet.switch(
            aet.eq(self.input.shape[1], aet.constant(self.batch_size)),
            self.batch_norm,
            self.full_norm,
        )


class ResidualLayer:
    def __init__(self, layers: list):
        """Definition of the Residual layer block

        Args:
            layers (list): a list of layers that defines the residual block

        Example:

            .. code-block:: python

                res_layer = ResidualLayer(layers=[
                    DenseLayer(w_1, b_1, activation=relu),
                    DenseLayer(w_2, b_2, activation=relu)
                ])
        """
        for layer in layers:
            if not isinstance(layer, Layer):
                raise TypeError(f"{layer} is not a Layer class instance")

        self.layers = layers
        self.params = []
        self._updates = []

    def apply(self, input):
        """Function to apply the input to the computational graph"""
        if isinstance(input, (list, tuple)):
            input = aet.stack(input)
        self.input = input

        for n, layer in enumerate(self.layers):
            if n == 0:
                layer.apply(self.input)
            else:
                layer.apply(self.layers[n - 1].output)
            self.params.extend(layer.params)
            self._updates.extend(layer.updates)
        self._output = self.layers[-1].output + self.input

    @property
    def updates(self):
        """Returns a list of update tuple pairs"""
        return self._updates

    def output(self):
        """Returns the output of this layer"""
        return self._output
