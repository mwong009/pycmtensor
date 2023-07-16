# functions.py
"""PyCMTensor functions module"""
from ctypes import util
from typing import Union

import aesara
import aesara.tensor as aet
import aesara.tensor.nlinalg as nlinalg
import numpy as np
from aesara.tensor.sharedvar import TensorSharedVariable
from aesara.tensor.var import TensorVariable

from pycmtensor.expressions import Beta, Expressions, Param
from pycmtensor.logger import error, log

__all__ = [
    "exp_mov_average",
    "logit",
    "loglikelihood",
    "rmse",
    "mae",
    "kl_divergence",
    "kl_multivar_norm",
    "errors",
    "second_order_derivative",
    "first_order_derivative",
]


def exp_mov_average(
    batch_avg: TensorVariable, moving_avg: TensorVariable, alpha: float = 0.1
):
    """Calculates the exponential moving average (EMA) of a new minibatch

    Args:
        batch_avg: the new batch value of the mean
        moving_avg: the moving value of the accumulated mean
        alpha: the moving average factor of the batch mean

    Returns:
        (TensorVariable): the new moving average

    Note:
        The moving average will decay by the difference between the existing value
        and the new value multiplied by the moving average factor. A higher `alpha`
        value results in faster changing moving average.

        Formula:

        $$x_{EMA} = \\alpha * x_t + x_{EMA} * (1-\\alpha)$$
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
        utility: list of M utility equations
        avail: list of M availability conditions,
            if no availability conditions are provided, defaults to `1` for all
            availabilities.

    Returns:
        (TensorVariable): A NxM matrix of probabilities.

    Note:
        The 0-th dimension is the numbering of alternatives, the N-th dimension is the size of the input (# rows).
    """

    if isinstance(utility, (list, tuple)):
        if (avail != None) and (len(utility) != len(avail)):
            msg = f"{utility} must have the same length as {avail}"
            raise ValueError(msg)
        _utility = utility
        for n, u in enumerate(_utility):
            # convert u to vector representation if u is a scalar
            if not isinstance(u, TensorVariable):
                utility[n] = aet.as_tensor_variable(u())

        # get maximum ndim from set of utlity equations
        max_ndim = max([u.ndim for u in utility])
        _utility = utility

        # pad tensors to the left to have the same number of dimensions
        utility = [aet.atleast_Nd(u, n=max_ndim) for u in _utility]

        # broadcast tensors across each other before stacking
        utility = aet.broadcast_arrays(*utility)

        # stack list of tensors to into a max_ndim+1 tensor
        U = aet.stack(utility)

    # use the utility inputs as is if given as a TensorVariable
    elif isinstance(utility, TensorVariable):
        U = utility
    else:
        raise NotImplementedError(
            f"utility {utility} has to be a list, tuple or TensorVariable instance"
        )

    # calculate the probabilities
    prob = aet.special.softmax(U, axis=0)

    # update probabilities by availability conditions
    if avail != None:
        # stack availabilities and convert to tensor
        AV = aet.stack(avail)

        while AV.ndim < prob.ndim:
            # insert new axes after choice axis (choice axis = 0)
            AV = aet.shape_padaxis(AV, axis=1)

        prob = prob * AV

        # normalize probabilities to sum to 1
        prob = prob / aet.sum(prob, axis=0, keepdims=1)

    return prob


def log_likelihood(prob: TensorVariable, y: TensorVariable):
    """Symbolic representation of the log likelihood cost function.

    Args:
        prob: a `TensorVariable` matrix describing the choice probabilites
        y: a `TensorVariable` referencing the choice variable

    Returns:
        (TensorVariable): a symbolic representation of the log likelihood with `ndim=0`.

    Note:
        The 0-th dimension is the numbering of alternatives, the N-th dimension is the size of the input (# rows).
    """
    # calculate the log probabilitiy of axis 0
    logprob = aet.log(prob)[y, ..., aet.arange(y.shape[0])]

    # take the average over all other non choice, non-n axes
    while logprob.ndim > 1:
        logprob = aet.mean(logprob, axis=1)

    return aet.sum(logprob)


def rmse(y_hat: TensorVariable, y: TensorVariable):
    """Computes the root mean squared error (RMSE) between pairs of observations

    Args:
        y_hat: model estimated values
        y: ground truth values

    Returns:
        (TensorVariable): symbolic scalar representation of the rmse

    Note:
        Tensor is flattened to a `dim=1` vector if the input tensor is `dim=2`.

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
        y_hat: model estimated values
        y : ground truth values

    Returns:
        (TensorVariable): symbolic scalar representation of the mean absolute error

    Note:
        Tensor is flattened to a `dim=1` vector if the input tensor is `dim=2`.
    """
    if y_hat.ndim != y.ndim:
        msg = f"y_hat should have the same dimensions as y. y_hat.ndim: {y_hat.ndim}, q.ndim: {y.ndim}"
        raise ValueError(msg)
    if y_hat.ndim < 1:
        y_hat = aet.flatten(y_hat)
        y = aet.flatten(y)

    return aet.mean(aet.abs(y_hat - y))


def kl_divergence(p: TensorVariable, q: TensorVariable):
    """Computes the KL divergence loss between discrete distributions `p` and `q`.

    Args:
        p: model output probabilities
        q: ground truth probabilities

    Returns:
        (TensorVariable): a symbolic representation of the KL loss 

    Note:
        Formula:

        $$
        L = \\begin{cases}
            \\sum_{i=1}^N (p_i * log(p_i/q_i)) & p>0\\\\
            0 & p<=0
        \\end{cases}
        $$
        
    """
    if p.ndim != q.ndim:
        msg = f"p should have the same shape as q. p.ndim: {p.ndim}, q.ndim: {q.ndim}"
        raise ValueError(msg)

    return aet.sum(aet.switch(aet.neq(p, 0), p * aet.log(p / q), 0))


def kl_multivar_norm(m0, v0, m1, v1, epsilon=1e-6):
    """Computes the KL divergence loss between two multivariate normal distributions.

    Args:
        m0 (TensorVariable): mean vector of the first Normal m.v. distribution $N_0$
        v0 (TensorVariable): (co-)variance matrix of the first Normal m.v. distribution
            $N_0$
        m1 (TensorVariable): mean vector of the second Normal m.v. distribution $N_1$
        v1 (TensorVariable): (co-)variance of the second Normal m.v. distribution $N_1$
        epsilon (float): small value to prevent divide-by-zero error

    Note:
        k = dimension of the distribution.

        Formula:

        $$
            D_{KL}(N_0||N_1) = 0.5 * \\Big(\\ln\\big(\\frac{|v_1|}{|v_0|}\\big) + trace(v_1^{-1} v_0) + (m_1-m_0)^T v_1^{-1} (m_1-m_0) - k\\Big)
        $$

        In variational inference, the kl divergence is the relative entropy between a
        diagonal multivariate Normal and a standard Normal distribution, $N(0, 1)$,
        therefore, for VI, `m1=1`, `v1=1`

        For two univariate distributions, dimensions of `m0,m1,v0,v1 = 0`
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
    """Symbolic representation of the discrete prediction as a percentage error.

    Args:
        prob: matrix describing the choice probabilites
        y: the `TensorVariable` referencing the choice column

    Returns:
        (TensorVariable): the mean prediction error over `y`
    """
    pred = aet.argmax(prob, axis=0)

    if y.dtype.startswith("int"):
        return aet.mean(aet.neq(pred, y))
    else:
        raise NotImplementedError(f"y should be int32 or int64", ("y.dtype:", y.dtype))


def second_order_derivative(cost: TensorVariable, params: list[Beta]):
    """Symbolic representation of the 2nd order Hessian matrix given cost.

    Args:
        cost: the cost function to compute the gradients over
        params: list of params to compute the gradients over

    Returns:
        (TensorVariable): the Hessian matrix of the cost function wrt to the params

    Note:
        Parameters with `status=1` are ignored.
    """
    if not isinstance(params, list):
        raise TypeError(f"params is not list instance. type(params)={type(params)}")
    params = [p() for p in params if (p.status != 1)]
    grads = aet.grad(cost, params, disconnected_inputs="ignore")
    mat = aet.as_tensor_variable(np.zeros((len(grads), len(grads))))
    for i in range(len(grads)):
        mat = aet.set_subtensor(
            x=mat[i, :],
            y=aet.grad(grads[i], params, disconnected_inputs="ignore"),
        )
    return mat


def first_order_derivative(cost: TensorVariable, params: list[Beta]):
    """Symbolic representation of the 1st order gradient vector given the cost.

    Args:
        cost: the cost function to compute the gradients over
        params: list of params to compute the gradients over

    Returns:
        (TensorVariable): the gradient vector of the cost function wrt to the params

    Note:
        Parameters with `status=1` are ignored.
    """
    if not isinstance(params, (dict, list)):
        raise TypeError(
            f"params is not list or dict instance. type(params)={type(params)}"
        )
    if isinstance(params, dict):
        params = list(params.values())
    params = [p() for p in params if (p.status != 1)]
    grads = aet.grad(cost, params, disconnected_inputs="ignore")
    return aet.as_tensor_variable(grads)
