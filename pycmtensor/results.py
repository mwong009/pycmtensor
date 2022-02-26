# results.py

import aesara
import dill as pickle
import numpy as np
import pandas as pd
from numpy import nan_to_num as nan2num

from pycmtensor.functions import bhhh, hessians
from pycmtensor.statistics import *


class Results:
    def __init__(self, model, database, show_weights=False):
        """Generate the output results to stdout

        Args:
            model (Class): The model class object or pickle file
            db (Class): The database class object
        """
        if isinstance(model, str):
            self.name = model
            with open(self.name, "rb") as f:
                model = pickle.load(f)
        else:
            self.name = model.name

        sample_size = len(database.data)
        n_params = len(model.get_betas())
        n_weights = model.get_weight_size()
        k = n_params + n_weights

        null_loglike = model.null_ll
        max_loglike = model.best_ll

        self.beta_results = get_beta_statistics(model, database)
        self.rho_square = nan2num(1.0 - max_loglike / null_loglike)
        self.rho_square_bar = nan2num(1.0 - (max_loglike - k) / null_loglike)
        self.ll_ratio_test = -2.0 * (null_loglike - max_loglike)

        self.akaike = 2.0 * (k - max_loglike)
        self.bayesian = -2.0 * max_loglike + k * np.log(sample_size)
        self.g_norm = gradient_norm(model, database)
        self.correlationMatrix = correlation_matrix(model, database)
        self.model = model
        self.n_params = n_params
        self.n_weights = n_weights
        self.sample_size = sample_size
        self.excluded_data = database.excludedData
        self.best_epoch = model.best_epoch
        self.best_ll_score = model.best_ll_score
        self.null_loglike = null_loglike
        self.max_loglike = max_loglike

        self.print_results = (
            f"Results for model {self.name}\n"
            + f"Number of Beta parameters: {self.n_params}\n"
            + (
                f"Total size of Neural Net weights: {self.n_weights}\n"
                if n_weights > 0
                else ""
            )
            + f"Sample size: {self.sample_size}\n"
            + f"Excluded data: {self.excluded_data}\n"
            + f"Init loglikelihood: {self.null_loglike:.3f}\n"
            + f"Final loglikelihood: {self.max_loglike:.3f}\n"
            + f"Likelihood ratio test: {self.ll_ratio_test:.3f}\n"
            + f"Accuracy: {(100 * self.best_ll_score):.3f}%\n"
            + f"Rho square: {self.rho_square:.3f}\n"
            + f"Rho bar square: {self.rho_square_bar:.3f}\n"
            + f"Akaike Information Criterion: {self.akaike:.2f}\n"
            + f"Bayesian Information Criterion: {self.bayesian:.2f}\n"
            + f"Final gradient norm: {self.g_norm:.3f}\n"
        )

        print(self.print_results)
        print(self.beta_results.to_string(), "\n")

        if model.get_weight_size() != 0:
            self.weights = model.output_estimated_weights()
            if show_weights:
                for w, value in zip(model.get_weights(), self.weights):
                    random = "zeros"
                    if w.random_init:
                        random = "random"
                    print(f"{w.name} {w.size} init: {random}\n{value}\n")
        else:
            self.weights = None

    def __str__(self):
        return self.print_results + self.beta_results.to_string()


class Predict:
    def __init__(self, model, database):
        """Class to output estimated model predictions

        Usage:
            Call Predict(model, database).probs() or .choices() for probabilities or
            discrete choices (argmax) respectively

        Args:
            model (PyCMTensor): the estimated model class object
            database (pycmtensor.Database): the given database object
        """
        self.model = model
        self.database = database
        self.columns = None
        if hasattr(database, "choices"):
            self.columns = database.choices

    def probs(self):
        db = self.database
        data_obj = self.model.output_probabilities().T
        return pd.DataFrame(data_obj, columns=self.columns)

    def choices(self):
        db = self.database
        data_obj = self.model.output_choices().T
        return pd.DataFrame(data_obj, columns=[db.choiceVar])


def get_beta_statistics(model, db):
    H = aesara.function(
        inputs=[],
        outputs=hessians(model.p_y_given_x, model.y, model.beta_params),
        on_unused_input="ignore",
        givens={t: data for t, data in zip(model.inputs, db.input_shared_data())},
    )

    BHHH = aesara.function(
        inputs=[],
        outputs=bhhh(model.p_y_given_x, model.y, model.beta_params),
        on_unused_input="ignore",
        givens={t: data for t, data in zip(model.inputs, db.input_shared_data())},
    )

    h = H()
    bh = BHHH()

    pandas_stats = pd.DataFrame(
        columns=[
            "Value",
            "Std err",
            "t-test",
            "p-value",
            "Rob. Std err",
            "Rob. t-test",
            "Rob. p-value",
        ],
        index=[p.name for p in model.beta_params if (p.status != 1)],
    )
    stderr = stderror(h, model.beta_params)
    robstderr = rob_stderror(h, bh, model.beta_params)
    pandas_stats["Std err"] = stderr
    pandas_stats["t-test"] = t_test(stderr, model.beta_params)
    pandas_stats["p-value"] = p_value(stderr, model.beta_params)
    pandas_stats["Rob. Std err"] = robstderr
    pandas_stats["Rob. t-test"] = t_test(robstderr, model.beta_params)
    pandas_stats["Rob. p-value"] = p_value(robstderr, model.beta_params)
    pandas_stats = pd.DataFrame(index=[p.name for p in model.beta_params]).join(
        pandas_stats
    )

    pandas_stats["Value"] = np.asarray(model.output_estimated_betas())

    return np.round(pandas_stats, 6).sort_index().fillna("-").astype("O")
