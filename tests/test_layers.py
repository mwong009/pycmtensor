# test_layers.py
import aesara
import numpy as np
import pytest
from aesara.tensor import tanh
from aesara.tensor.nnet import relu

from pycmtensor.expressions import Bias, Param, Weight
from pycmtensor.models.layers import BatchNormLayer, DenseLayer


def test_denselayer(swissmetro_db):
    layers = []
    params = []
    updates = []
    db = swissmetro_db
    input = db[
        [
            "PURPOSE",
            "FIRST",
            "TICKET",
            "WHO",
            "LUGGAGE",
            "AGE",
            "MALE",
            "INCOME",
            "GA",
            "ORIGIN",
            "DEST",
        ]
    ]

    rng = np.random.default_rng()
    w_1 = Weight("w_1", size=(11, 32), init_type="he", rng=rng)
    b_1 = Bias("b_1", size=(32,))

    layers.append(DenseLayer(input=input, w=w_1, bias=b_1))
    params.extend(layers[-1].params)
    updates.extend(layers[-1].updates)

    w_2 = Weight("w_2", size=(32, 3), init_type="glorot", rng=rng)
    b_2 = Bias("b_2", size=(3,))

    layers.append(DenseLayer(input=layers[-1].output(), w=w_2, bias=b_2))
    params.extend(layers[-1].params)
    updates.extend(layers[-1].updates)


def test_batch_norm_layer(swissmetro_db):
    layers = []
    params = []
    updates = []
    db = swissmetro_db
    input = db[
        [
            "PURPOSE",
            "FIRST",
            "TICKET",
            "WHO",
            "LUGGAGE",
            "AGE",
            "MALE",
            "INCOME",
            "GA",
            "ORIGIN",
            "DEST",
        ]
    ]

    rng = np.random.default_rng()
    bn_gamma = Param("bn_gamma", value=np.ones(11))
    bn_beta = Param("bn_beta", value=np.zeros(11))
    layers.append(BatchNormLayer(input, bn_gamma, bn_beta, batch_size=32))
    params.extend(layers[-1].params)
    updates.extend(layers[-1].updates)

    f = aesara.function(
        inputs=db.x,
        outputs=layers[-1].output().T,
        updates=updates,
    )

    g = aesara.function(
        inputs=db.x,
        outputs=layers[-1].output().T,
    )

    for i in range(100):
        out = f(*db.pandas.inputs(db.x, rng.integers(0, 333), 32, 0))

    assert len(layers[0].mv_mean.eval()) == 11
    assert len(layers[0].mv_var.eval()) == 11
    assert len(layers[0].updates) == 2

    out = g(*db.pandas.inputs(db.x))
    assert out.shape == (10719, 11)
