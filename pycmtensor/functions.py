# functions.py
"""PyCMTensor functions module"""
import aesara.tensor as aet
import aesara.tensor.nlinalg as nlinalg
import numpy as np

from .logger import log


def logit(utility: list, avail: list = None):
    """Computes the Logit function, with availability conditions.

    Args:
        utility (list): List of M utility equations.
        avail (list): List of M availability conditions. If no availabilities are
            provided, defaults to 1 for all availabilities.

    Returns:
        TensorVariable: A NxM matrix of probabilities.

    Note:
        The 0-th dimension is the numbering of alternatives.
    """

    if isinstance(utility, (list, tuple)):
        if (avail != None) and (len(utility) != len(avail)):
            msg = f"{utility} must have the same length as {avail}"
            raise ValueError(msg)
        U = aet.stack(utility)
    else:
        U = utility

    prob = aet.nnet.softmax(U, axis=0)
    if avail != None:
        AV = aet.stack(avail)
        while AV.ndim < prob.ndim:
            AV = aet.expand_dims(AV, 1)
        prob = prob * AV
        prob = prob / aet.sum(prob, axis=0, keepdims=1)
    return prob


def log_likelihood(prob, y):
    """Symbolic representation of the log likelihood cost function.

    Args:
        prob (TensorVariable): matrix describing the choice probabilites
        y (TensorVariable): ``TensorVariable`` referencing the choice column

    Returns:
        TensorVariable: a symbolic representation of the log likelihood with ndim=0.

    Note:
        The 0-th dimension is the numbering of alternatives.
    """
    return aet.sum(aet.log(prob)[y, ..., aet.arange(y.shape[0])])


def kl_divergence(p, q):
    """Computes the KL divergence loss between discrete distributions ``p`` and ``q``.

    Args:
        p (TensorVariable): Output probabilities
        q (TensorVariable): Reference probabilities

    Returns:
        TensorVariable: a symbolic representation of the KL loss with ndim=0.

    Notes:
        L = $\sum$ [y_prob * log(y_prob/prob) where y_prob>0, else 0]
    """
    if p.ndim != q.ndim:
        msg = f"p should have the same shape as q. p.ndim: {p.ndim}, q.ndim: {q.ndim}"
        log(40, msg)
        raise ValueError(msg)
    return aet.sum(aet.switch(aet.neq(p, 0), p * aet.log(p / q), 0))


def kl_multivar_norm(m0, v0, m1, v1):
    """Computes the KL divergence loss between two multivariate normal distributions.

    Args:
        m0: mean vector of the first Normal m.v. distribution $N_0$
        v0: (co-)variance matrix of the first Normal m.v. distribution $N_0$
        m1: mean vector of the second Normal m.v. distribution $N_1$
        v1: (co-)variance of the second Normal m.v. distribution $N_1$

    Notes:
        If m0 and v0 are 0 and 1 respectively, returns a univariate norm solution.
        If m1 and v1 are 0 and 1 respectively, computes a simplified version of the multivariate norm.

        k = dimension of the distribution.

        Formula: $$
        D_{KL}(N_0||N_1) = 0.5 * (ln(|v_1|/|v_0|) + trace(v_1^{-1} * v_0) +
            (m_1-m_0)^T * v_1^{-1} * (m_1-m_0) - k)
        $$

    """
    if not (
        (m0.ndim >= m1.ndim)
        and (v0.ndim >= v1.ndim)
        and (m0.ndim <= 1)
        and (v0.ndim <= 2)
    ):
        msg = f"Incorrect dimensions inputs: m0.ndim={m0.ndim}, v0.ndim={v0.ndim}, m1.ndim={m1.ndim}, v1.ndim={v1.ndim}"
        log(40, msg)
        raise ValueError(msg)

    if (m0.ndim == v0.ndim == 0) or (m1.ndim == v1.ndim == 0):
        # computes univariate norm or multivariate norm with N(m1, v1)=N(0, 1)
        if v0.ndim == 2:
            v0 = aet.diag(v0)
        return aet.sum(0.5 * ((v0 + aet.sqr(m0 - m1)) / v1 - aet.log(v0 / v1) - 1))

    k = m0.shape[0]
    v1_inv = nlinalg.inv(v1)
    det_term = aet.log(nlinalg.det(v1) / nlinalg.det(v0))
    trace_term = nlinalg.trace(aet.dot(v1_inv, v0))
    return aet.sum(
        det_term + trace_term + aet.dot((m1 - m0).T, aet.dot(v1_inv, (m1 - m0))) - k
    )


def errors(prob, y):
    """Symbolic representation of the prediction as a percentage error.

    Args:
        prob (TensorVariable): matrix describing the choice probabilites
        y (TensorVariable): The ``TensorVariable`` referencing the choice column

    Returns:
        TensorVariable: the mean prediction error over the input ``y``
    """
    pred = aet.argmax(prob, axis=0)

    if y.dtype.startswith("int"):
        return aet.mean(aet.neq(pred, y))
    else:
        raise NotImplementedError(f"y should be int32 or int64", ("y.dtype:", y.dtype))


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
