# statistics.py
"""PyCMTensor statistics module"""
import aesara.tensor.nlinalg as nlinalg
import numpy as np
from numpy import nan_to_num as nan2num
from scipy import stats

__all__ = [
    "correlation_matrix",
    "p_value",
    "rob_correlation_matrix",
    "rob_stderror",
    "stderror",
    "t_test",
]


def variance_covariance(h):
    """Returns the variance covariance matrix given the Hessian (``h``)"""
    return nlinalg.pinv(nan2num(-h)).eval()


def rob_variance_covariance(h, bh):
    """Returns the rob. var-covar matrix given the Hessian (``h``) and the BHHH
    matrix (``bh``)
    """
    cr = variance_covariance(h)
    return cr.dot(bh.dot(cr))


def t_test(stderr, params):
    """Returns the t stats of ``params`` given the standard error (``stderr``)"""
    params = [value for key, value in params.items()]
    return [p.mean() / s for p, s in zip(params, stderr)]


def p_value(stderr, params):
    """Returns the p-value of ``params`` given the standard error (``stderr``)"""
    tTest = t_test(stderr, params)
    return [2.0 * (1.0 - stats.norm.cdf(abs(t))) for t in tTest]


def stderror(h, params):
    """Returns the standard error of ``params`` given the Hessian (``h``)

    The std err is calculated as the square root of the variance covariance
    matrix.

    """
    # params = [p for p in params if (p.status != 1)]
    varCovar = variance_covariance(h)
    stdErr = []
    for i, p in enumerate(params):
        if (params[p].shape == ()) and (params[p] == 0.0):
            stdErr.append(np.nan)
        elif varCovar[i, i] < 0:
            stdErr.append(np.finfo(float).max)
        else:
            stdErr.append(np.sqrt(varCovar[i, i] + 1e-8))  # for numerical stability

    return stdErr


def rob_stderror(h, bh, params):
    """Returns the rob. standard error of ``params`` given the Hessian (``h``) and the
    BHHH matrix (``bh``)
    """
    # params = [p for p in params if (p.status != 1)]
    robVarCovar = rob_variance_covariance(h, bh)
    robstderr = []
    for i, p in enumerate(params):
        if (params[p].shape == ()) and (params[p] == 0.0):
            robstderr.append(np.nan)
        elif robVarCovar[i, i] < 0:
            robstderr.append(np.finfo(float).max)
        else:
            robstderr.append(np.sqrt(robVarCovar[i, i] + 1e-8))

    return robstderr


def correlation_matrix(h):
    """Returns the correlation matrix given the Hessian (``h``)"""
    var_covar = variance_covariance(h)
    d = np.diag(var_covar)
    if (d > 0).all():
        diag = np.diag(np.sqrt(d))
        diag_inv = nlinalg.inv(diag).eval()
        mat = diag_inv.dot(var_covar.dot(diag_inv))
    else:
        mat = np.full_like(var_covar, np.finfo(float).max)

    return mat


def rob_correlation_matrix(h, bh):
    """Returns the correlation matrix given the Hessian and the BHHH matrix"""
    rob_var_covar = rob_variance_covariance(h, bh)
    rd = np.diag(rob_var_covar)
    if (rd > 0).all():
        diag = np.diag(np.sqrt(rd))
        diag_inv = nlinalg.inv(diag).eval()
        mat = diag_inv.dot(rob_var_covar.dot(diag_inv))
    else:
        mat = np.full_like(rob_var_covar, np.finfo(float).max)

    return mat
