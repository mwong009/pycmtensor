# functions.py
"""PyCMTensor functions module"""
import aesara.tensor as aet
import aesara.tensor.nlinalg as nlinalg
import numpy as np

from pycmtensor import log


def logit(utility: list, avail: list = None):
    """Computes the Logit function, with availability conditions.

    Args:
        utility (list): List of M utility equations.
        avail (list): List of M availability conditions. If no availabilities are
        provided, defaults to 1 for all availabilities.

    Returns:
        TensorVariable: A NxM matrix of probabilities.
    """

    if isinstance(utility, (list, tuple)):
        if (avail != None) and (len(utility) != len(avail)):
            msg = f"{utility} must have the same length as {avail}"
            raise ValueError(msg)
        U = aet.stack(utility).flatten(2)
    else:
        U = utility

    prob = aet.nnet.softmax(U, axis=0)
    if avail != None:
        AV = aet.stack(avail)
        assert U.ndim == AV.ndim
        prob *= AV
        prob = prob / aet.sum(prob, axis=0, keepdims=1)
    return prob


def log_likelihood(prob, y):
    """Symbolic representation of the log likelihood cost function.

    Args:
        prob (TensorVariable): Matrix describing the choice probabilites.
        y (TensorVariable): `TensorVariable`` referencing the choice column.

    Returns:
        TensorVariable: a symbolic representation of the log likelihood with ndim=0.
    """
    return aet.sum(aet.log(prob)[y, aet.arange(y.shape[0])])


def kl_divergence(p, q):
    """Computes the KL divergence loss between discrete distributions ``p`` and ``q``.

    Args:
        p (TensorVariable): Output probabilities
        q (TensorVariable): Reference probabilities

    Returns:
        TensorVariable: a symbolic representation of the KL loss with ndim=0.

    Notes:
        loss = \sum [y_prob * log(y_prob/prob) where y_prob>0, else 0]
    """
    if p.ndim != q.ndim:
        msg = f"p should have the same shape as q. p.ndim: {p.ndim}, q.ndim: {q.ndim}"
        log(40, msg)
        raise ValueError(msg)
    return aet.sum(aet.switch(aet.neq(p, 0), p * aet.log(p / q), 0))


def kl_univar_norm(m0: float, v0: float, m1: float, v1: float):
    """Computes the KL divergence loss between two univariate normal distributions.

    Args:
        m0 (scalar): mean of the first Normal distribution $N_0$
        v0 (scalar): variance of the first Normal distribution $N_0$
        m1 (scalar): mean of the second Normal distribution $N_1$
        v1 (scalar): variance of the second Normal distribution $N_1$

    Notes:
        Formula: $$
        D_{KL}(N_0||N_1) = -log(sqrt(v_0/v_1)) + (v_0 + (m_0-m_1)^2)/(2 * v_1) - 0.5
        $$

        If m1=0 and v1=1, then $$
        D_{KL}(N_0||N_1) = -log(sqrt(v_0)) + (v_0 + m_0^2)/2 - 0.5
        $$
    """
    if (v0 < 0) or (v1 < 0):
        msg = f"Invalid input value for variance v1 ({v1}) in kl_univar_norm()"
        log(40, msg)
        raise ValueError(msg)
    return 0.5 * ((v0 + aet.sqr(m0 - m1)) / v1 - aet.log(v0 / v1) - 1)


def errors(prob, y):
    """Symbolic representation of the prediction as a percentage error.

    Args:
        prob (TensorVariable): Matrix describing the choice probabilites.
        y (TensorVariable): The ``TensorVariable`` referencing the choice column.

    Raises:
        TypeError: ``y`` should have the same shape as ``pred``.
        NotImplementedError: ``y`` should be an ``int`` Type.

    Returns:
        TensorVariable: a symbolic representation of the prediction error with ndim=0.
    """
    if prob.ndim > 2:
        prob = aet.reshape(prob, [prob.shape[0], y.shape[0], -1])
        prob = aet.mean(prob, axis=-1)
    pred = aet.argmax(prob, axis=0)

    if y.ndim != pred.ndim:
        msg = f"y should have the same shape as pred. y.ndim: {y.ndim}, pred.ndim: {pred.ndim}"
        log(40, msg)
        raise ValueError(msg)
    if y.dtype.startswith("int"):
        return aet.mean(aet.neq(pred, y))
    else:
        raise NotImplementedError(f"y should be an int Type", ("y.dtype:", y.dtype))


def hessians(ll, params):
    """Symbolic representation of the Hessian matrix given the log likelihood.

    Args:
        ll (TensorVariable): the loglikelihood to compute the gradients over
        params (list): list of params to compute the gradients over

    Returns:
        TensorVariable: the Hessian matrix with ndim=2

    Note:
        Parameters with status=1 are ignored.
    """
    if not isinstance(params, (dict, list)):
        raise TypeError(
            f"params is not list or dict instance. type(params)={type(params)}"
        )
    if isinstance(params, dict):
        params = list(params.values())
    params = [p() for p in params if (p.status != 1)]
    grads = aet.grad(ll, params, disconnected_inputs="ignore")
    mat = aet.as_tensor_variable(np.zeros((len(grads), len(grads))))
    for i in range(len(grads)):
        mat = aet.set_subtensor(
            x=mat[i, :],
            y=aet.grad(grads[i], params, disconnected_inputs="ignore"),
        )
    return mat


def bhhh(ll, params):
    """Symbolic representation of the Berndt-Hall-Hall-Hausman (BHHH) algorithm given
    the log likelihood.

    Args:
        ll (TensorVariable): the loglikelihood to compute the gradients over
        params (list): list of params to compute the gradients over

    Returns:
        TensorVariable: the outer product of the gradient with ndim=2

    Note:
        Parameters with status=1 are ignored.
    """
    if not isinstance(params, (dict, list)):
        raise TypeError(
            f"params is not list or dict instance. type(params)={type(params)}"
        )
    if isinstance(params, dict):
        params = list(params.values())
    params = [p() for p in params if (p.status != 1)]
    grads = aet.grad(ll, params, disconnected_inputs="ignore")
    mat = aet.outer(aet.as_tensor_variable(grads), aet.as_tensor_variable(grads).T)
    return mat


def gnorm(cost, params):
    """Symbolic representation of the gradient norm given the cost.

    Args:
        cost (TensorVariable): the cost to compute the gradients over
        params (list): list of params to compute the gradients over

    Returns:
        TensorVariable: the gradient norm value

    Note:
        Parameters with status=1 are ignored.
    """
    if not isinstance(params, (dict, list)):
        raise TypeError(
            f"params is not list or dict instance. type(params)={type(params)}"
        )
    if isinstance(params, dict):
        params = list(params.values())
    params = [p() for p in params if (p.status != 1)]
    grads = aet.grad(cost, params, disconnected_inputs="ignore")
    return nlinalg.norm(grads, ord=None)
