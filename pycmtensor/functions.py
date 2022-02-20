# functions.py

import aesara.tensor as aet
import numpy as np


def logit(utility, avail):
    """Logit function, with availability conditions

    Args:
        utility (list): List of utility equations (TensorVariable)
        avail (list): List of availability cond. (TensorVariable)

    Returns:
        TensorVariable: A NxN probability vectors corresponding to the choice index
    """
    if isinstance(utility, (list, tuple)):
        assert len(utility) == len(
            avail
        ), f"{utility} must have the same length as {avail}"
        U = aet.stacklists(utility).flatten(2)
    else:
        U = utility

    AV = aet.stacklists(avail)

    assert U.ndim == AV.ndim

    prob = aet.nnet.softmax(U, axis=0) * AV
    return prob / aet.sum(prob, axis=0, keepdims=1)


def neg_loglikelihood(prob, y):
    nll = -aet.mean(aet.log(prob)[y, aet.arange(y.shape[0])])
    return nll


def full_loglikelihood(prob, y):
    ll = -neg_loglikelihood(prob, y) * y.shape[0]
    # ll = aet.sum(aet.log(prob)[y, aet.arange(y.shape[0])])
    return ll


def errors(prob, y):
    # symbolic description of how to compute prediction as a class
    if prob.ndim > 2:
        prob = aet.reshape(prob, [prob.shape[0], y.shape[0], -1])
        prob = aet.mean(prob, axis=-1)
    pred = aet.argmax(prob, axis=0)

    if y.ndim != pred.ndim:
        raise TypeError(
            "y should have the same shape as pred", ("y", y.type, "pred", pred.type)
        )
    if y.dtype.startswith("int"):
        return aet.mean(aet.neq(pred, y))
    else:
        raise NotImplementedError()


def hessians(prob, y, params):
    params = [p() for p in params if (p.status != 1)]
    for p in params:
        assert p.ndim == 0, f"{p.name}, ndim={p.ndim}"
    grads = aet.grad(full_loglikelihood(prob, y), params, disconnected_inputs="ignore")
    _h = aet.as_tensor_variable(np.zeros((len(grads), len(grads))))
    for i in range(len(grads)):
        _h = aet.set_subtensor(
            x=_h[i, :], y=aet.grad(grads[i], params, disconnected_inputs="ignore"),
        )
    return _h


def bhhh(prob, y, params):
    params = [p() for p in params if (p.status != 1)]
    for p in params:
        assert p.ndim == 0, f"{p.name}, ndim={p.ndim}"
    grads = aet.grad(full_loglikelihood(prob, y), params, disconnected_inputs="ignore")
    _bh = aet.outer(aet.as_tensor_variable(grads), aet.as_tensor_variable(grads).T)
    return _bh
