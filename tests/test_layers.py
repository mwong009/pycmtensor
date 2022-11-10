# test_layers.py
import aesara
import numpy as np
import pytest
from aesara.tensor.math import sigmoid, tanh
from aesara.tensor.nnet import relu

from pycmtensor.expressions import Bias, Param, Weight
from pycmtensor.models.layers import BatchNormLayer, DenseLayer


class TestLayers:
    @pytest.fixture
    def rng(self, config_class):
        return config_class.rng

    def test_denselayer(self, rng, swissmetro_db):
        layers = []
        params = []
        updates = []
        db = swissmetro_db
        input = db[["PURPOSE", "FIRST", "TICKET", "WHO", "AGE", "MALE", "GA", "ORIGIN"]]

        w_1 = Weight(name="w_1", size=(8, 32), init_type="he", rng=rng)
        b_1 = Bias(name="b_1", size=(32,))
        layers.append(DenseLayer(w=w_1, bias=b_1))
        params.extend(layers[-1].params)
        updates.extend(layers[-1].updates)

        w_2 = Weight(name="w_1", size=(32, 3), init_type="glorot", rng=rng)
        b_2 = Bias(name="b_1", size=(3,))
        layers.append(DenseLayer(w=w_2, bias=b_2))
        params.extend(layers[-1].params)
        updates.extend(layers[-1].updates)

        w_3 = Weight(name="w_3", size=(3, 3), init_type="zeros")
        b_3 = Bias(name="b_3", size=(3,))
        layers.append(DenseLayer(w=w_3, bias=b_3))
        params.extend(layers[-1].params)
        updates.extend(layers[-1].updates)

        assert layers[0].activation == relu
        assert layers[1].activation == tanh
        assert layers[2].activation == sigmoid

        for n, layer in enumerate(layers):
            if n == 0:
                layer.apply(input)
            else:
                layer.apply(layers[n - 1].output)

        output = layers[-1].output
        assert len(updates) == 3
        assert len(params) == 6

        f = aesara.function(inputs=db.x, outputs=output.swapaxes(0, 1))

        a = f(*db.pandas.inputs(tensors=db.x, index=2, batch_size=32))
        assert a.shape == (32, 3)

    def test_batchnormlayer(self, rng, swissmetro_db):
        layers = []
        params = []
        updates = []
        db = swissmetro_db
        input = db[["PURPOSE", "FIRST", "TICKET", "WHO", "AGE", "MALE", "GA", "ORIGIN"]]

        bn_gamma = Param("bn_gamma", value=np.ones(8))
        bn_beta = Param("bn_beta", value=np.zeros(8))
        layers.append(BatchNormLayer(gamma=bn_gamma, beta=bn_beta, batch_size=32))
        layers[-1].apply(input)
        params.extend(layers[-1].params)
        updates.extend(layers[-1].updates)

        for n, layer in enumerate(layers):
            if n == 0:
                layer.apply(input)
            else:
                layer.apply(layers[n - 1].output)

        f = aesara.function(
            inputs=db.x,
            outputs=layers[-1].output.T,
            updates=updates,
        )

        g = aesara.function(
            inputs=db.x,
            outputs=layers[-1].output.T,
        )

        for _ in range(100):
            out = f(*db.pandas.inputs(db.x, rng.integers(0, 333), 32))

        assert len(layers[0].mv_mean.eval()) == 8
        assert len(layers[0].mv_var.eval()) == 8
        assert len(updates) == 2
        assert len(params) == 2

        out = g(*db.pandas.inputs(db.x))
        assert out.shape == (8575, 8)
