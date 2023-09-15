# statistics.py
"""PyCMTensor statistics module

This module contains methods for calculating the statistics of the estimated parameters.
"""
import numpy as np
from scipy import stats

__all__ = [
    "correlation_matrix",
    "rob_correlation_matrix",
    "p_value",
    "rob_stderror",
    "stderror",
    "t_test",
]


def variance_covariance(hessian):
    """computes the variance covariance matrix given the Hessian

    Args:
        hessian (numpy.ndarray): a 2-D hessian matrix

    Returns:
        (numpy.ndarray): the variance covariance matrix


    !!! notes

        The variance covariance matrix is calculated by taking the inverse of the (negative) hessian matrix. If the inverse is undefined, returns a zero or a large finite number.

        $$
        varcovar = -H^{-1}
        $$
    """
    return np.linalg.pinv(np.nan_to_num(-hessian))


def rob_variance_covariance(hessian, bhhh):
    """computes the robust variance covariance matrix given the Hessian and the BHHH matrices

    Args:
        hessian (numpy.ndarray): the hessian matrix
        bhhh (numpy.ndarray): the BHHH matrix

    Returns:
        (numpy.ndarray): the robust variance covariance matrix

    !!! notes

        The robust variance covariance matrix is computed as follows:

        $$
        rob. varcovar = (-H)^{-1}\\cdot BHHH\\cdot (-H)^{-1}
        $$
    """
    cr = variance_covariance(hessian)
    return cr.dot(bhhh.dot(cr))


def t_test(stderr, params):
    """computes the statistical t-test of the estimated parameter and the standard error

    Args:
        stderr (list): standard errors
        params (list): estimated parameters

    Returns:
        (list): t-test of the estimated parameters
    """
    params = [value for _, value in params.items()]
    return [np.percentile(p, 50) / s for p, s in zip(params, stderr)]


def p_value(stderr, params):
    """computes the p-value (statistical significance) of the estimated parameter using the two-tailed normal distribution, where p-value=$2(1-\\phi(|t|)$, $\\phi$ is the cdf of the normal distribution

    Args:
        stderr (list): standard errors
        params (list): estimated parameters

    Returns:
        (list): p-value of the estimated parameters
    """
    ttest = t_test(stderr, params)
    return [2.0 * (1.0 - stats.norm.cdf(abs(t))) for t in ttest]


def stderror(hessian, params):
    """calculates the standard error of the estimated parameter given the hessian matrix

    Args:
        hessian (numpy.ndarray): the hessian matrix
        params (list): estimated parameters

    Returns:
        (list): the standard error of the estimates

    !!! note

        The standard errors is calculated using the formula:

        $$
        stderr = \\sqrt{diag(-H^{-1})}
        $$
    """
    varCovar = variance_covariance(hessian)
    stdErr = []
    for i, p in enumerate(params):
        if (params[p].shape == ()) and (params[p] == 0.0):
            stdErr.append(np.nan)
        elif varCovar[i, i] < 0:
            stdErr.append(np.finfo(float).max)
        else:
            stdErr.append(np.sqrt(varCovar[i, i] + 1e-8))  # for numerical stability

    return stdErr


def rob_stderror(hessian, bhhh, params):
    """calculates the robust standard error of the estimated parameter given the hessian and the bhhh matrices

    Args:
        hessian (numpy.ndarray): the hessian matrix
        bhhh (numpy.ndarray): the bhhh matrix
        params (list): estimated parameters

    Returns:
        (list): the robust standard error of the estimates

    !!! note

        The robust standard errors is calculated using the formula:

        $$
        rob. stderr = \\sqrt{diag(rob. varcovar)}
        $$
    """
    robVarCovar = rob_variance_covariance(hessian, bhhh)
    robstderr = []
    for i, p in enumerate(params):
        if (params[p].shape == ()) and (params[p] == 0.0):
            robstderr.append(np.nan)
        elif robVarCovar[i, i] < 0:
            robstderr.append(np.finfo(float).max)
        else:
            robstderr.append(np.sqrt(robVarCovar[i, i] + 1e-8))

    return robstderr


def correlation_matrix(hessian):
    """computes the correlation matrix from the hessian matrix

    Args:
        hessian (numpy.ndarray): the hessian matrix

    Returns:
        (numpy.ndarray): the correlation matrix
    """
    var_covar = variance_covariance(hessian)
    d = np.diag(var_covar)
    if (d > 0).all():
        diag = np.diag(np.sqrt(d))
        diag_inv = np.linalg.inv(diag)
        mat = diag_inv.dot(var_covar.dot(diag_inv))
    else:
        mat = np.full_like(var_covar, np.finfo(float).max)

    return mat


def rob_correlation_matrix(hessian, bhhh):
    """computes the robust correlation matrix from the hessian and bhhh matrix

    Args:
        hessian (numpy.ndarray): the hessian matrix
        bhhh (numpy.ndarray): the bhhh matrix

    Returns:
        (numpy.ndarray): the tobust correlation matrix
    """
    rob_var_covar = rob_variance_covariance(hessian, bhhh)
    rd = np.diag(rob_var_covar)
    if (rd > 0).all():
        diag = np.diag(np.sqrt(rd))
        diag_inv = np.linalg.inv(diag)
        mat = diag_inv.dot(rob_var_covar.dot(diag_inv))
    else:
        mat = np.full_like(rob_var_covar, np.finfo(float).max)

    return mat
