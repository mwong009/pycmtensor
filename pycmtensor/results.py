# results.py
"""PyCMTensor results module

This module provides the results of the estimated model and output formatting
"""
import numpy as np
import pandas as pd
from numpy import nan_to_num as nan2num

from pycmtensor.logger import error
from pycmtensor.statistics import *

__all__ = ["Results"]


class Results(object):
    def __init__(self):
        """Base class object to hold model results and scores

        Attributes:
            build_time (str): string formatted time stamp of the duration of the build
                stage
            train_time (str): string formatted time stamp of the duration of the
                training stage
            epochs_per_sec (float): number of epochs of the training dataset per
                second, benchmark for calculation speed
            n_params (int): total number of parameters, used for calculating statistics
            n_train (int): number of training samples used
            n_valid (int): number of validation samples used
            seed (int): random seed value used
            null_loglikelihood (float): null log likelihood of the model
            best_loglikelihood (float): the final estimated model log likelihood
            best_valid_error (float): final estimated model validation error
            best_epoch (int): the epoch number at the final estimated model
            gnorm (float): the gradient norm at the final estimated model
            hessian_matrix (numpy.ndarray): the 2-D hessian matrix
            bhhh_matrix (numpy.ndarray): the 3-D bhhh matrix where the 1st dimension is
                the length of the dataset and the last 2 dimensions are the matrix for
                each data observation
            statistics_graph (dict): a dictionary containing the learning_rate, training, and validation statistics
            betas (dict): a dictionary containing the Beta coefficients
            params (dict): a dictionary containing all model coefficients
        """
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
        self.best_train_error = 1.0
        self.best_epoch = 0

        self.gnorm = None
        self.hessian_matrix = None
        self.bhhh_matrix = None

        self.betas = {}
        self.params = {}

        self.statistics_graph = None

    def rho_square(self):
        """rho square value of the model

        Returns:
            (float): analogue for the model fit
        """
        return nan2num(1.0 - self.best_loglikelihood / self.null_loglikelihood)

    def rho_square_bar(self):
        """McFadden's adjusted rho square

        Returns:
            (float): the adjusted McFadden's rho square value
        """
        k = self.n_params
        return nan2num(1.0 - (self.best_loglikelihood - k) / self.null_loglikelihood)

    def loglikelihood_ratio_test(self):
        """Log likelihood ratio test

        Returns:
            (float): the log likelihood ratio test: $-2 \\times (NLL-LL)
        """
        return -2.0 * (self.null_loglikelihood - self.best_loglikelihood)

    def AIC(self):
        """Akaike information criterion

        Returns:
            (float): the AIC of the model
        """
        k = self.n_params
        return 2.0 * (k - self.best_loglikelihood)

    def BIC(self):
        """Bayesian information criterion, adjusted for the number of parameters and number of training samples

        Returns:
            (float): the BIC of the model
        """
        k = self.n_params
        n = self.n_train
        return -2.0 * self.best_loglikelihood + k * np.log(n)

    def benchmark(self):
        """benchmark statistics

        Returns:
            (pandas.DataFrame): Summary of the model performance benchmark
        """
        stats = pd.DataFrame(columns=["value"])
        stats.loc["Seed"] = self.config.seed
        stats.loc["Model build time"] = self.build_time
        stats.loc["Model train time"] = self.train_time
        stats.loc["epochs per sec"] = f"{self.epochs_per_sec} epoch/s"
        return stats

    def model_statistics(self):
        """model statistics

        Returns:
            (pandas.DataFrame): Summary of the model statistics
        """
        stats = pd.DataFrame(columns=["value"]).astype("object")
        stats.loc["Number of training samples used"] = f"{self.n_train:d}"
        stats.loc["Number of validation samples used"] = f"{self.n_valid:d}"
        stats.loc["Number of estimated parameters in the model"] = f"{self.n_params:d}"
        stats.loc["Null. log likelihood"] = f"{self.null_loglikelihood:.2f}"
        stats.loc["Final log likelihood"] = f"{self.best_loglikelihood:.2f}"
        stats.loc["Validation Accuracy"] = f"{100*(1-self.best_valid_error):.2f}%"
        stats.loc["Training Accuracy"] = f"{100*(1-self.best_train_error):.2f}%"
        stats.loc["Likelihood ratio test"] = f"{self.loglikelihood_ratio_test():.2f}"
        stats.loc["Rho square"] = f"{self.rho_square():.3f}"
        stats.loc["Rho square bar"] = f"{self.rho_square_bar():.3f}"
        stats.loc["Akaike Information Criterion"] = f"{self.AIC():.2f}"
        stats.loc["Bayesian Information Criterion"] = f"{self.BIC():.2f}"
        stats.loc["Final gradient norm"] = f"{self.gnorm:.3e}"
        stats.loc[
            "Maximum epochs reached"
        ] = f"{'Yes' if self.best_epoch == self.config.max_epochs else 'No'}"
        stats.loc["Best result at epoch"] = f"{self.best_epoch:d}"
        return stats

    def beta_statistics(self):
        """Beta statistics

        Returns:
            (pandas.DataFrame): Summary of the estimated Beta statistics
        """
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
            data=[np.percentile(value, 50) for _, value in self.betas.items()],
            columns=["value"],
        ).round(3)

        stats["std err"] = stderror(h, self.betas)
        stats["t-test"] = t_test(stats["std err"], self.betas)
        stats["p-value"] = p_value(stats["std err"], self.betas)

        stats["rob. std err"] = rob_stderror(h, bh, self.betas)
        stats["rob. t-test"] = t_test(stats["rob. std err"], self.betas)
        stats["rob. p-value"] = p_value(stats["rob. std err"], self.betas)
        stats = stats.round(3)

        for key, value in self.betas.items():
            if value.shape != ():
                stats.at[key + " (sd)", "value"] = round(value.std(), 3)

        stats = stats.sort_index().fillna("-").astype("O")
        return stats

    def model_correlation_matrix(self):
        """Correlation matrix calculated from the hessian

        Returns:
            (pandas.DataFrame): The correlation matrix
        """
        if self.hessian_matrix.ndim > 2:
            h = self.hessian_matrix.sum(axis=0)
        else:
            h = self.hessian_matrix

        mat = pd.DataFrame(
            columns=self.betas,
            index=self.betas,
            data=correlation_matrix(h),
        )

        return mat.round(3)

    def model_robust_correlation_matrix(self):
        """Robust correlation matrix calculated from the hessian and bhhh

        Returns:
            (pandas.DataFrame): The robust correlation matrix
        """
        if self.hessian_matrix.ndim > 2:
            h = self.hessian_matrix.sum(axis=0)
        else:
            h = self.hessian_matrix

        if self.bhhh_matrix.ndim > 2:
            bh = self.bhhh_matrix.sum(axis=0)
        else:
            bh = self.bhhh_matrix

        mat = pd.DataFrame(
            columns=self.betas,
            index=self.betas,
            data=rob_correlation_matrix(h, bh),
        )

        return mat.round(3)

    def show_training_plot(self, sample_intervals=1):
        """Displays the statistics graph as a line plot

        Args:
            sample_intervals (int): plot only the n-th point from the statistics

        Returns:
            (pandas.DataFrame): The statistics as a pandas dataframe
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style("white")
        color = sns.color_palette("tab10", 4)

        statistics = pd.DataFrame(data=self.statistics_graph)
        if (sample_intervals < 1) or (sample_intervals > len(statistics)):
            error(f"sample_intervals must be greater than 0 or less than total length")
            return statistics

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        statistics = statistics[::sample_intervals]
        sns.lineplot(
            data=statistics["train_error"] * 100,
            ax=ax1,
            color=color[0],
            label="Train Error",
        )
        sns.lineplot(
            data=statistics["valid_error"] * 100,
            ax=ax1,
            color=color[1],
            label="Valid Error",
        )
        sns.lineplot(
            data=statistics["train_ll"],
            color=color[2],
            ax=ax2,
            label="Train Loglikelihood",
        )

        ax1.set(ylabel="Error (%)", xlabel="Epoch")
        ax1.legend(loc="upper right", bbox_to_anchor=(0.4, -0.1))
        ax2.set(ylabel="Loglikelihood")
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(2, 3))
        ax2.legend(loc="upper left", bbox_to_anchor=(0.6, -0.1))
        plt.tight_layout()
        plt.show()

        return statistics
