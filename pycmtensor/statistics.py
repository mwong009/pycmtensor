# statistics.py

import aesara
import aesara.tensor as aet
import numpy as np
import pandas as pd
from numpy import nan_to_num as nan2num
from scipy import linalg, stats

from pycmtensor.functions import neg_loglikelihood


def variance_covariance(h):
    return -linalg.pinv(nan2num(h))


def rob_variance_covariance(h, bhhh):
    varCovar = variance_covariance(h)
    return varCovar.dot(bhhh.dot(varCovar))


def t_test(stderr, params):
    params = [p() for p in params if (p.status != 1)]
    return [p.eval() / s for p, s in zip(params, stderr)]


def p_value(stderr, params):
    tTest = t_test(stderr, params)
    return [2.0 * (1.0 - stats.norm.cdf(abs(t))) for t in tTest]


def stderror(h, params):
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


def correlation_matrix(model):
    h = model.H()

    varCovar = variance_covariance(h)
    d = np.diag(varCovar)
    if (d > 0).all():
        diag = np.diag(np.sqrt(d))
        diagInv = linalg.inv(diag)
        matrix = diagInv.dot(varCovar.dot(diagInv))
    else:
        matrix = np.full_like(varCovar, np.finfo(float).max)

    df = pd.DataFrame(
        columns=[p.name for p in model.beta_params if (p.status != 1)],
        index=[p.name for p in model.beta_params if (p.status != 1)],
        data=matrix,
    )
    return df


def rob_correlation_matrix(model):
    robVarCovar = rob_variance_covariance(h=model.H(), bhhh=model.BH())
    rd = np.diag(robVarCovar)
    if (rd > 0).all():
        diag = np.diag(np.sqrt(rd))
        diagInv = linalg.inv(diag)
        matrix = diagInv.dot(robVarCovar.dot(diagInv))
    else:
        matrix = np.full_like(robVarCovar, np.finfo(float).max)

    df = pd.DataFrame(
        columns=[p.name for p in model.beta_params if (p.status != 1)],
        index=[p.name for p in model.beta_params if (p.status != 1)],
        data=matrix,
    )
    return df
