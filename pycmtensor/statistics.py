# statistics.py
"""PyCMTensor statistics module"""
import aesara.tensor as aet
import aesara.tensor.nlinalg as nlinalg
import numpy as np
from aesara import function
from numpy import nan_to_num as nan2num
from scipy import stats

__all__ = [
    "correlation_matrix",
    "p_value",
    "rob_correlation_matrix",
    "rob_stderror",
    "stderror",
    "t_test",
    "elasticities",
]


def variance_covariance(h):
    """Returns the var-covar matrix given the Hessian (``h``)"""
    return -nlinalg.pinv(nan2num(h)).eval()


def rob_variance_covariance(h, bhhh):
    """Returns the rob. var-covar matrix given the Hessian (``h``) and the BHHH
    matrix (``bhhh``)
    """
    var_covar = variance_covariance(h)
    return var_covar.dot(bhhh.dot(var_covar))


def t_test(stderr, params):
    """Returns the t stats of ``params`` given the standard error (``stderr``)"""
    params = [p() for p in params if (p.status != 1)]
    return [p.eval() / s for p, s in zip(params, stderr)]


def p_value(stderr, params):
    """Returns the p-value of ``params`` given the standard error (``stderr``)"""
    tTest = t_test(stderr, params)
    return [2.0 * (1.0 - stats.norm.cdf(abs(t))) for t in tTest]


def stderror(h, params):
    """Returns the standard error of ``params`` given the Hessian (``h``)"""
    params = [p() for p in params if (p.status != 1)]
    varCovar = variance_covariance(h)
    stdErr = []
    for i in range(len(params)):
        if varCovar[i, i] < 0:
            stdErr.append(np.finfo(float).max)
        else:
            stdErr.append(np.sqrt(varCovar[i, i] + 1e-8))  # for numerical stability

    return stdErr


def rob_stderror(h, bhhh, params):
    """Returns the rob. standard error of ``params`` given the Hessian (``h``) and the
    BHHH matrix (``bhhh``)
    """
    params = [p() for p in params if (p.status != 1)]
    varCovar = variance_covariance(h)
    robVarCovar = varCovar.dot(bhhh.dot(varCovar))
    robstderr = []
    for i in range(len(params)):
        if robVarCovar[i, i] < 0:
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


def rob_correlation_matrix(h, bhhh):
    """Returns the correlation matrix given the Hessian and the BHHH matrix"""
    rob_var_covar = rob_variance_covariance(h, bhhh)
    rd = np.diag(rob_var_covar)
    if (rd > 0).all():
        diag = np.diag(np.sqrt(rd))
        diag_inv = nlinalg.inv(diag).eval()
        mat = diag_inv.dot(rob_var_covar.dot(diag_inv))
    else:
        mat = np.full_like(rob_var_covar, np.finfo(float).max)

    return mat


def elasticities(model, db, y: int, x: str):
    """Returns the disaggregate point elasticities of choice y wrt x

    Args:
        model (PyCMTensorModel): the model class object
        db (pycmtensor.Data): the database object
        y (int): the alternative index
        x (str): the name of the variable to derive the elasticities

    Returns:
        list: the disaggregate point elasticity E_n
    """
    fn_elasticity = function(
        inputs=model.inputs,
        outputs=aet.grad(
            aet.sum(model.p_y_given_x[y]), db[x], disconnected_inputs="ignore"
        )
        * db[x]
        / model.p_y_given_x[y],
        on_unused_input="ignore",
    )
    data = db.pandas.inputs(model.inputs, split_type="train")
    return fn_elasticity(*data)
