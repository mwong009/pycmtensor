# results.py
"""PyCMTensor results module"""
import numpy as np
import pandas as pd
from numpy import nan_to_num as nan2num

from .statistics import *

__all__ = ["Results"]


class Results:
    """Class object to hold model results"""

    def __init__(self):
        self.build_time = None
        self.train_time = None
        self.iterations_per_sec = None
        self.n_params = None
        self.n_train_samples = None
        self.n_valid_samples = None
        self.seed = None

        self.null_loglikelihood = -np.inf
        self.init_loglikelihood = -np.inf
        self.best_loglikelihood = -np.inf
        self.best_valid_error = 1.0
        self.best_step = 0

        self.gnorm = None
        self.hessian_matrix = None
        self.bhhh_matrix = None
        self.betas = None
        self.weights = None

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
        n = self.n_train_samples
        return -2.0 * self.best_loglikelihood + k * np.log(n)

    def benchmark(self):
        """Returns a pandas DataFrame of a summary of the model benchmark"""
        stats = pd.DataFrame(columns=["value"])
        stats.loc["Seed"] = self.seed
        stats.loc["Model build time"] = self.build_time
        stats.loc["Model train time"] = self.train_time
        stats.loc["iterations per sec"] = f"{self.iterations_per_sec}/s"
        return stats

    def model_statistics(self):
        """Returns a pandas DataFrame of a summary of the model training"""
        stats = pd.DataFrame(columns=["value"]).astype("object")
        stats.loc["Number of training samples used"] = int(self.n_train_samples)
        stats.loc["Number of validation samples used"] = int(self.n_valid_samples)
        stats.loc["Init. log likelihood"] = self.init_loglikelihood
        stats.loc["Final log likelihood"] = self.best_loglikelihood
        stats.loc["Accuracy"] = f"{100*(1-self.best_valid_error):.2f}%"
        stats.loc["Likelihood ratio test"] = self.loglikelihood_ratio_test()
        stats.loc["Rho square"] = self.rho_square()
        stats.loc["Rho square bar"] = self.rho_square_bar()
        stats.loc["Akaike Information Criterion"] = self.AIC()
        stats.loc["Bayesian Information Criterion"] = self.BIC()
        stats.loc["Final gradient norm"] = self.gnorm
        return stats

    def beta_statistics(self):
        """Returns a pandas DataFrame of the model beta statistics"""
        betas = self.betas
        h = self.hessian_matrix
        bhhh = self.bhhh_matrix

        stats = pd.DataFrame(
            index=[b.name for b in betas if (b.status != 1)], columns=["value"]
        )
        stats["std err"] = stderror(h, betas)
        stats["t-test"] = t_test(stats["std err"], betas)
        stats["p-value"] = p_value(stats["std err"], betas)

        stats["rob. std err"] = rob_stderror(h, bhhh, betas)
        stats["rob. t-test"] = t_test(stats["rob. std err"], betas)
        stats["rob. p-value"] = p_value(stats["rob. std err"], betas)
        stats.drop("value", axis=1, inplace=True)

        df = pd.DataFrame(
            data=[b().eval() for b in betas],
            index=[b.name for b in betas],
            columns=["value"],
        )
        stats = pd.concat([df, stats], axis=1).sort_index().fillna("-").astype("O")

        return stats

    def model_correlation_matrix(self):
        """Returns a pandas DataFrame of the model correlation matrix"""
        betas = self.betas
        h = self.hessian_matrix

        mat = pd.DataFrame(
            columns=[b.name for b in betas if (b.status != 1)],
            index=[b.name for b in betas if (b.status != 1)],
            data=correlation_matrix(h),
        )

        return mat

    def model_robust_correlation_matrix(self):
        """Returns a pandas DataFrame of the model (robust) correlation matrix"""
        betas = self.betas
        h = self.hessian_matrix
        bhhh = self.bhhh_matrix

        mat = pd.DataFrame(
            columns=[b.name for b in betas if (b.status != 1)],
            index=[b.name for b in betas if (b.status != 1)],
            data=rob_correlation_matrix(h, bhhh),
        )

        return mat
