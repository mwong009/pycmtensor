# results.py
"""PyCMTensor results module"""
import numpy as np
import pandas as pd
from numpy import nan_to_num as nan2num

import pycmtensor.models.layers as layers
from pycmtensor.expressions import Beta
from pycmtensor.statistics import *

__all__ = ["Results"]


class Results:
    """Class object to hold model results"""

    def __init__(self):
        self.build_time = None
        self.train_time = None
        self.epochs_per_sec = None
        self.n_params = None
        self.n_train = None
        self.n_valid = None
        self.seed = None

        self.null_loglikelihood = -np.inf
        self.best_loglikelihood = -np.inf
        self.best_valid_error = 1.0
        self.best_step = 0

        self.gnorm = None
        self.hessian_matrix = None
        self.bhhh_matrix = None
        self.betas = None

        self.performance_graph = None
        self.lr_history_graph = None

    def rho_square(self):
        """Returns the rho square statistics"""
        return nan2num(1.0 - self.best_loglikelihood / self.null_loglikelihood)

    def rho_square_bar(self):
        """Returns the rho square bar statistics"""
        k = self.n_params
        return nan2num(1.0 - (self.best_loglikelihood - k) / self.null_loglikelihood)

    def loglikelihood_ratio_test(self):
        """Returns the log likelihood ratio test statistics"""
        return -2.0 * (self.null_loglikelihood - self.best_loglikelihood)

    def AIC(self):
        """Returns the Akaike Information Criterion"""
        k = self.n_params
        return 2.0 * (k - self.best_loglikelihood)

    def BIC(self):
        """Returns the Bayesian Information Criterion"""
        k = self.n_params
        n = self.n_train
        return -2.0 * self.best_loglikelihood + k * np.log(n)

    def benchmark(self):
        """Returns a pandas DataFrame of a summary of the model benchmark"""
        stats = pd.DataFrame(columns=["value"])
        stats.loc["Seed"] = self.seed
        stats.loc["Model build time"] = self.build_time
        stats.loc["Model train time"] = self.train_time
        stats.loc["epochs per sec"] = f"{self.epochs_per_sec} iter/s"
        return stats

    def model_statistics(self):
        """Returns a pandas DataFrame of a summary of the model training"""
        stats = pd.DataFrame(columns=["value"]).astype("object")
        stats.loc["Number of training samples used"] = int(self.n_train)
        stats.loc["Number of validation samples used"] = int(self.n_valid)
        stats.loc["Null. log likelihood"] = self.null_loglikelihood
        stats.loc["Final log likelihood"] = self.best_loglikelihood
        stats.loc["Accuracy"] = f"{100*(1-self.best_valid_error):.2f}%"
        stats.loc["Likelihood ratio test"] = self.loglikelihood_ratio_test()
        stats.loc["Rho square"] = self.rho_square()
        stats.loc["Rho square bar"] = self.rho_square_bar()
        stats.loc["Akaike Information Criterion"] = self.AIC()
        stats.loc["Bayesian Information Criterion"] = self.BIC()
        stats.loc["Final gradient norm"] = f"{self.gnorm:.5e}"
        return stats

    def beta_statistics(self):
        """Returns a pandas DataFrame of the model beta statistics"""
        n = len(self.hessian_matrix)
        if self.hessian_matrix.ndim > 2:
            h = self.hessian_matrix.sum(axis=0)
        else:
            h = self.hessian_matrix

        if self.bhhh_matrix.ndim > 2:
            bh = self.bhhh_matrix.sum(axis=0)
        else:
            bh = self.bhhh_matrix

        stats = pd.DataFrame(
            index=self.betas,
            data=[value.mean() for _, value in self.betas.items()],
            columns=["value"],
        )

        stats["std err"] = stderror(h, self.betas)
        stats["t-test"] = t_test(stats["std err"], self.betas)
        stats["p-value"] = p_value(stats["std err"], self.betas)

        stats["rob. std err"] = rob_stderror(h, bh, self.betas)
        stats["rob. t-test"] = t_test(stats["rob. std err"], self.betas)
        stats["rob. p-value"] = p_value(stats["rob. std err"], self.betas)

        for key, value in self.betas.items():
            if value.shape != ():
                stats.at[key + " (sd)", "value"] = value.std()

        stats = stats.sort_index().fillna("-").astype("O")
        return stats

    def model_correlation_matrix(self):
        """Returns a pandas DataFrame of the model correlation matrix"""
        h = self.hessian_matrix.sum(axis=0)

        mat = pd.DataFrame(
            columns=self.betas,
            index=self.betas,
            data=correlation_matrix(h),
        )

        return mat

    def model_robust_correlation_matrix(self):
        """Returns a pandas DataFrame of the model (robust) correlation matrix"""
        h = self.hessian_matrix.sum(axis=0)
        bh = self.bhhh_matrix.sum(axis=0)

        mat = pd.DataFrame(
            columns=self.betas,
            index=self.betas,
            data=rob_correlation_matrix(h, bh),
        )

        return mat
