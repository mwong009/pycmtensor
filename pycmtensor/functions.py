# functions.py
"""PyCMTensor functions module"""
from typing import Union

import aesara.tensor as aet
import aesara.tensor.nlinalg as nlinalg
import numpy as np
from aesara.tensor.var import TensorVariable

from pycmtensor.expressions import Beta

from .logger import error, log


def exp_mov_average(
    batch_avg: TensorVariable, moving_avg: TensorVariable, alpha: float = 0.1
):
    """Calculates the exponential moving average (EMA) of a new minibatch

    Args:
        batch_avg (TensorVariable): the new batch value of the mean
        moving_avg (TensorVariable): the moving value of the accumulated mean
        alpha (float): the moving average factor of the batch mean

    Returns:
        TensorVariable: the new moving average

    Note:
        The moving average will decay by the difference between the existing value
        and the new value multiplied by the moving average factor. A higher ``alpha``
        value results in faster changing moving average.

        Formula:

        .. math::

            x_{EMA} = \\alpha * x_t + x_{EMA} * (1-\\alpha)
    """

    while moving_avg.ndim < batch_avg.ndim:
        moving_avg = aet.expand_dims(moving_avg, -1)

    ema = batch_avg * alpha + moving_avg * (1 - alpha)
    return ema


def logit(
    utility: Union[list, tuple, TensorVariable],
    avail: Union[list, tuple, TensorVariable] = None,
):
    """Computes the Logit function, with availability conditions.

    Args:
        utility (list, tuple, TensorVariable): list of :math:`M` utility equations
        avail (list, tuple, TensorVariable): list of :math:`M` availability conditions,
            if no availability conditions are provided, defaults to 1 for all
            availabilities.

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


def log_likelihood(prob: TensorVariable, y: TensorVariable):
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


def rmse(y_hat: TensorVariable, y: TensorVariable):
    """Computes the root mean squared error (RMSE) between pairs of observations

    Args:
        y_hat (TensorVariable): model estimated values
        y (TensorVariable): ground truth values

    Returns:
        TensorVariable: symbolic scalar representation of the rmse

    Note:
        Tensor is flattened to an N-vector if the input args are :math:`N\\times 1`
        matrices.

        Formula:

        .. math::

            RMSE = \\sqrt{\\frac{1}{N}\\sum_{i=1}^N(\\hat{y}_i-y_i)^2}

    """
    if y_hat.ndim != y.ndim:
        msg = f"y_hat should have the same dimensions as y. y_hat.ndim: {y_hat.ndim}, q.ndim: {y.ndim}"
        raise ValueError(msg)
    if y_hat.ndim < 1:
        y_hat = aet.flatten(y_hat)
        y = aet.flatten(y)

    return aet.sqrt(aet.mean(aet.sqr(y_hat - y)))


def mae(y_hat: TensorVariable, y: TensorVariable):
    """Computes the mean absolute error (MAE) between pairs of observations

    Args:
        y_hat (TensorVariable): model estimated values
        y (TensorVariable): ground truth values

    Returns:
        TensorVariable: symbolic scalar representation of the mae

    Note:
        Tensor is flattened to an N-vector if the input args are :math:`N\\times 1` matrices.

        Formula:

        .. math::

            MAE = \\frac{\sum_{i=1}^N|\\hat{y}_i-y_i|}{N}
    """
    if y_hat.ndim != y.ndim:
        msg = f"y_hat should have the same dimensions as y. y_hat.ndim: {y_hat.ndim}, q.ndim: {y.ndim}"
        raise ValueError(msg)
    if y_hat.ndim < 1:
        y_hat = aet.flatten(y_hat)
        y = aet.flatten(y)

    return aet.mean(aet.abs(y_hat - y))


def kl_divergence(p: TensorVariable, q: TensorVariable):
    """Computes the KL divergence loss between discrete distributions ``p`` and ``q``.

    Args:
        p (TensorVariable): model output probabilities
        q (TensorVariable): ground truth probabilities

    Returns:
        TensorVariable: a symbolic representation of the KL loss with

    Note:
        Formula:

        .. math::

            L = \\begin{cases}
                \\sum_{i=1}^N (p_i * log(p_i/q_i)) & p>0\\\\
                0 & p<=0
            \\end{cases}
    """
    if p.ndim != q.ndim:
        msg = f"p should have the same shape as q. p.ndim: {p.ndim}, q.ndim: {q.ndim}"
        raise ValueError(msg)

    return aet.sum(aet.switch(aet.neq(p, 0), p * aet.log(p / q), 0))


def kl_multivar_norm(m0, v0, m1, v1, epsilon=1e-6):
    """Computes the KL divergence loss between two multivariate normal distributions.

    Args:
        m0: mean vector of the first Normal m.v. distribution :math:`N_0`
        v0: (co-)variance matrix of the first Normal m.v. distribution :math:`N_0`
        m1: mean vector of the second Normal m.v. distribution :math:`N_1`
        v1: (co-)variance of the second Normal m.v. distribution :math:`N_1`
        epsilon (float): small value to prevent divide-by-zero error

    Note:
        k = dimension of the distribution.

        Formula:

        .. math::

            D_{KL}(N_0||N_1) = 0.5 * \\Big(\\ln\\big(\\frac{|v_1|}{|v_0|}\\big) + trace(v_1^{-1} v_0) + (m_1-m_0)^T v_1^{-1} (m_1-m_0) - k\\Big)

        In variational inference, the kl divergence is the relative entropy between a
        diagonal multivariate Normal and a standard Normal distribution, N(0, 1),
        therefore, for VI, ``m1=aet.constant(0)``, ``v1=aet.constant(1)``

        For two univariate distributions, dimensions of m0,m1,v0,v1 = 0
    """
    if not (
        (m0.ndim >= m1.ndim)
        and (v0.ndim >= v1.ndim)
        and (m0.ndim <= 1)
        and (v0.ndim <= 2)
    ):
        msg = f"Incorrect dimensions inputs: m0.ndim={m0.ndim}, v0.ndim={v0.ndim}, m1.ndim={m1.ndim}, v1.ndim={v1.ndim}"
        raise ValueError(msg)

    # VI KL divergence
    if (m1.ndim == v1.ndim == 0) and (m0.ndim == 1) and (v0.ndim == 2):
        v0 = aet.diag(v0)
        kld = 0.5 * aet.sum(v0 + aet.sqr(m0) - 1 - aet.log(v0))

    # univariate KL divergence
    elif (m0.ndim == v0.ndim == 0) and (m1.ndim == v1.ndim == 0):
        kld = (
            aet.log(aet.sqrt(v1 / (v0 + epsilon)))
            + 0.5 * (v0 + aet.sqr(m0 - m1)) / v1
            - 0.5
        )

    # multivariate KL divergence
    else:
        k = m0.shape[0]
        v1_inv = nlinalg.inv(v1)
        det_term = aet.log(nlinalg.det(v1) / nlinalg.det(v0))
        trace_term = nlinalg.trace(aet.dot(v1_inv, v0))
        prod_term = aet.dot((m1 - m0).T, aet.dot(v1_inv, (m1 - m0)))
        kld = aet.sum(det_term + trace_term + prod_term - k)

    return kld


def errors(prob: TensorVariable, y: TensorVariable):
    """Symbolic representation of the prediction as a percentage error.

    Args:
        prob (TensorVariable): matrix describing the choice probabilites
        y (TensorVariable): the ``TensorVariable`` referencing the choice column

    Returns:
        TensorVariable: the mean prediction error over the input ``y``
    """
    pred = aet.argmax(prob, axis=0)

    if y.dtype.startswith("int"):
        return aet.mean(aet.neq(pred, y))
    else:
        raise NotImplementedError(f"y should be int32 or int64", ("y.dtype:", y.dtype))


def hessians(ll: TensorVariable, params: list[Beta]):
    """Symbolic representation of the Hessian matrix given the log likelihood.

    Args:
        ll (TensorVariable): the loglikelihood to compute the gradients over
        params (list): list of params to compute the gradients over

    Returns:
        TensorVariable: the Hessian matrix

    Note:
        Parameters with status=1 are ignored.
    """
    if not isinstance(params, list):
        raise TypeError(f"params is not list instance. type(params)={type(params)}")
    params = [p() for p in params if (p.status != 1)]
    grads = aet.grad(ll, params, disconnected_inputs="ignore")
    mat = aet.as_tensor_variable(np.zeros((len(grads), len(grads))))
    for i in range(len(grads)):
        mat = aet.set_subtensor(
            x=mat[i, :],
            y=aet.grad(grads[i], params, disconnected_inputs="ignore"),
        )
    return mat


def bhhh(ll: TensorVariable, params: list[Beta]):
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


def gnorm(cost: TensorVariable, params: list[Beta]):
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
