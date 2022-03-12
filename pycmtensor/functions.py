# functions.py
"""Module containing tensor functions."""

import aesara.tensor as aet
import numpy as np
from scipy import linalg

from pycmtensor import logger as log


def logit(utility: list, avail: list = None):
    """Logit function, with availability conditions.

    Args:
        utility (list): List of utility equations.
        avail (list): List of availability conditions. If not availability is provided, defaults to 1 for all availabilities.

    Returns:
        TensorVariable: A NxN probability vectors corresponding to the choice index.
    """

    if isinstance(utility, (list, tuple)):
        if (avail != None) and (len(utility) != len(avail)):
            msg = f"{utility} must have the same length as {avail}"
            log.error(msg)
            raise ValueError(msg)
        U = aet.stacklists(utility).flatten(2)
    else:
        U = utility

    prob = aet.nnet.softmax(U, axis=0)
    if avail != None:
        AV = aet.stacklists(avail)
        assert U.ndim == AV.ndim
        prob *= AV
        prob = prob / aet.sum(prob, axis=0, keepdims=1)
    return prob


def neg_loglikelihood(prob, y):
    """Symbolic description of how to compute the average negative loglikelihood.

    Args:
        prob (TensorVariable): Function describing the choice probabilites.
        y (TensorVariable): The ``TensorVariable`` symbol referencing the choice column.

    Returns:
        TensorVariable: a symbolic description of the average negative loglikelihood.
    """
    nll = -(full_loglikelihood(prob, y) / y.shape[0])
    return nll


def full_loglikelihood(prob, y):
    """Symbolic description of how to compute the full loglikelihood.

    Args:
        prob (TensorVariable): Function describing the choice probabilites.
        y (TensorVariable): The ``TensorVariable`` symbol referencing the choice column.

    Returns:
        TensorVariable: a symbolic description of the full loglikelihood.
    """
    ll = aet.sum(aet.log(prob)[y, aet.arange(y.shape[0])])
    return ll


def errors(prob, y):
    """Symbolic description of how to compute prediction as a class.

    Args:
        prob (TensorVariable): Function describing the choice probabilites.
        y (TensorVariable): The ``TensorVariable`` symbol referencing the choice column.

    Raises:
        TypeError: ``y`` should have the same shape as ``pred``.
        NotImplementedError: ``y`` should be an ``int`` Type.

    Returns:
        TensorVariable: a symbolic description of the prediction error.
    """
    if prob.ndim > 2:
        prob = aet.reshape(prob, [prob.shape[0], y.shape[0], -1])
        prob = aet.mean(prob, axis=-1)
    pred = aet.argmax(prob, axis=0)

    if y.ndim != pred.ndim:
        msg = f"y should have the same shape as pred. y.ndim: {y.ndim}, pred.ndim: {pred.ndim}"
        log.error(msg)
        raise ValueError(msg)
    if y.dtype.startswith("int"):
        return aet.mean(aet.neq(pred, y))
    else:
        raise NotImplementedError(f"y should be an int Type", ("y.dtype:", y.dtype))


def hessians(prob, y, params):
    """Compute the hessians of the loglikelihood cost function.

    Args:
        prob (TensorVariable): Function describing the choice probabilites.
        y (TensorVariable): The ``TensorVariable`` symbol referencing the choice column.
        params (list): list of params to compute the gradients.

    Returns:
        Function: returns an Aesara ``function``.`` Calling the return value with no arguments computes the hessian function.

    Note:
        Parameters with status=1 are ignored.

        This is used as an internal function call.
    """
    params = [p() for p in params if (p.status != 1)]
    for p in params:
        if p.ndim != 0:
            msg = f"{p.name} is not a valid {p.ndim}-diamension Beta parameter."
            log.error(msg)
            raise ValueError(msg)
    grads = aet.grad(full_loglikelihood(prob, y), params, disconnected_inputs="ignore")
    _h = aet.as_tensor_variable(np.zeros((len(grads), len(grads))))
    for i in range(len(grads)):
        _h = aet.set_subtensor(
            x=_h[i, :],
            y=aet.grad(grads[i], params, disconnected_inputs="ignore"),
        )
    return _h


def bhhh(prob, y, params):
    """Compute the BHHH of the loglikelihood cost function.

    Args:
        prob (TensorVariable): Function describing the choice probabilites.
        y (TensorVariable): The ``TensorVariable`` symbol referencing the choice column
        params (list): list of params to compute the gradients.

    Returns:
        Function: returns an Aesara ``function``. Calling the return value with no arguments computes the BHHH function.

    Note:
        Parameters with status=1 are ignored.

        This is used as an internal function call.
    """
    params = [p() for p in params if (p.status != 1)]
    for p in params:
        if p.ndim != 0:
            msg = f"{p.name} is not a valid {p.ndim}-diamension Beta parameter."
            log.error(msg)
            raise ValueError(msg)
    grads = aet.grad(full_loglikelihood(prob, y), params, disconnected_inputs="ignore")
    _bh = aet.outer(aet.as_tensor_variable(grads), aet.as_tensor_variable(grads).T)
    return _bh


def gradient_norm(prob, y, params):
    """Compute the gradient norm of the cost function.

    Args:
        prob (TensorVariable): Function describing the choice probabilites.
        y (TensorVariable): The ``TensorVariable`` symbol referencing the choice column
        params (list): list of params to compute the gradients.

    Returns:
        Function: returns an Aesara ``function``. Calling the return value with no arguments computes the gradient norm.

    Note:
        Parameters with status=1 are ignored.

        This is used as an internal function call.
    """
    params = [p() for p in params if (p.status != 1)]
    for p in params:
        if p.ndim != 0:
            msg = f"{p.name} is not a valid {p.ndim}-diamension Beta parameter."
            log.error(msg)
            raise ValueError(msg)
    grads = aet.grad(neg_loglikelihood(prob, y), params, disconnected_inputs="ignore")
    norm = linalg.norm(grads)
    return norm
