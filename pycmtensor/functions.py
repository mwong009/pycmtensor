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
        U = aet.concatenate(utility, axis=1)
    else:
        U = utility

    AV = aet.concatenate(avail, axis=1)

    prob = aet.nnet.softmax(U, axis=1) * AV
    return prob / aet.sum(prob, axis=1, keepdims=1)


def neg_loglikelihood(prob, y):
    nll = -aet.mean(aet.log(prob)[aet.arange(y.shape[0]), y])
    return nll


def full_loglikelihood(prob, y):
    ll = aet.sum(aet.log(prob)[aet.arange(y.shape[0]), y])
    return ll


def errors(prob, y):
    # symbolic description of how to compute prediction as a class
    pred = aet.argmax(prob, axis=1)

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
    grads = aet.grad(full_loglikelihood(prob, y), params, disconnected_inputs="ignore")
    _h = aet.as_tensor_variable(np.zeros((len(grads), len(grads))))
    for i in range(len(grads)):
        _h = aet.set_subtensor(
            x=_h[i, :],
            y=aet.grad(grads[i], params, disconnected_inputs="ignore"),
        )
    return _h


def bhhh(prob, y, params):
    params = [p() for p in params if (p.status != 1)]
    grads = aet.grad(full_loglikelihood(prob, y), params, disconnected_inputs="ignore")
    _bh = aet.outer(aet.as_tensor_variable(grads), aet.as_tensor_variable(grads).T)
    return _bh
