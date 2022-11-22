# pycmtensor.py
"""PyCMTensor main module"""
from collections import OrderedDict
from time import perf_counter
from typing import Union

import dill as pickle
import numpy as np
from aesara import function
from aesara.tensor.var import TensorVariable

from pycmtensor import config, rng

from .expressions import Beta, ExpressionParser, Weight
from .functions import bhhh, errors, gnorm, hessians
from .logger import debug, info, warning
from .results import Results
from .utils import time_format


class PyCMTensorModel:
    def __init__(self, db):
        """Base model class object"""
        self.name = "PyCMTensorModel"
        self.config = config
        self.params = []  # keep track of all the params
        self.betas = []  # keep track of the Betas
        self.weights = []  # keep track of the Weights
        self.updates = []  # keep track of the updates
        self.inputs = db.all
        self.results = Results()

        debug(f"Building model...")

    def add_params(self, params: Union[dict, list]):
        """Method to load locally defined variables

        Args:
            params (Union[dict, list]): a dict or list of ``TensorSharedVariable``
        """
        if not isinstance(params, (dict, list)):
            raise TypeError(f"params must be of Type dict or list")

        # create a dict of str(param): param, if params is given as a list
        if isinstance(params, list):
            params = {str(p): p for p in params}

        # iterate through the dict of params:
        if not hasattr(self, "cost"):
            raise ValueError(f"No valid cost function defined.")

        symbols = ExpressionParser().parse(getattr(self, "cost"))
        seen = set()
        for _, p in params.items():
            if not isinstance(p, (Beta, Weight)):
                continue

            if p.name in seen:
                continue
                # raise NameError(f"Duplicate param names defined: {p.name}.")

            if (p.name not in symbols) and (p.status == 0):
                warning(f"Unused Beta {p.name} removed from computational graph")
                continue

            self.params.append(p)
            seen.add(p.name)

            if isinstance(p, (Beta)):
                self.betas.append(p)
            else:
                self.weights.append(p)

        self.n_params = self.get_n_params()

    def add_regularizers(self, l_reg: TensorVariable):
        """Adds regularizer to model cost function

        Args:
            l_reg (TensorVariable): symbolic variable defining the regularizer term
        """
        if not hasattr(self, "cost"):
            raise ValueError("No valid cost function defined.")

        self.cost += l_reg

    def get_n_params(self) -> int:
        """Get the total number of parameters"""
        return self.get_n_betas() + sum(self.get_n_weights())

    def get_n_betas(self) -> int:
        """Get the count of Beta parameters"""
        return len(self.betas)

    def get_betas(self) -> dict:
        """Returns the Beta (key, value) pairs as a dict"""
        return {beta.name: beta.get_value() for beta in self.betas}

    def get_n_weights(self) -> list[int]:
        """Get the total number of elements of each weight matrix"""
        return [w.size for w in self.get_weights()]

    def get_weights(self) -> list[np.ndarray]:
        """Returns the Weights as a list of matrices"""
        return [w.get_value() for w in self.weights]

    def reset_values(self):
        """Resets param values to their initial values"""
        for p in self.params:
            p.reset_value()

    def model_loglikelihood(self):
        """Loads the function to ``self.loglikelihood()`` to output the loglikelihood
        value of the model given inputs"""
        self.loglikelihood = function(
            name="loglikelihood",
            inputs=self.inputs,
            outputs=self.ll,
        )

    def model_choice_probabilities(self):
        """Loads the function to ``self.choice_probabilities()`` to output discrete
        choice probabilities. Axes of outputs are swapped"""
        self.choice_probabilities = function(
            name="choice probabilities",
            inputs=self.inputs,
            outputs=self.p_y_given_x.swapaxes(0, 1),
        )

    def model_choice_predictions(self):
        """Loads the function to ``self.choice_predictions()`` to output discrete
        choice predictions"""
        self.choice_predictions = function(
            name="choice predictions",
            inputs=self.inputs,
            outputs=self.pred,
        )

    def model_prediction_error(self):
        """Loads the function to ``self.prediction_error()`` to output the model error
        wrt inputs"""
        self.prediction_error = function(
            name="prediction error",
            inputs=self.inputs,
            outputs=errors(self.p_y_given_x, self.y),
        )

    def model_H(self):
        """Loads the function to ``self.H()`` to calculate the Hessian matrix or the
        2nd-order partial derivatives of the model.
        """
        self.H = function(
            name="Hessian matrix",
            inputs=self.inputs,
            outputs=hessians(self.ll, self.betas),
        )

    def model_BHHH(self):
        """Loads the function to ``self.BHHH()`` to calculate the Berndt-Hall-Hall-
        Hausman (BHHH) approximation.
        """
        self.BHHH = function(
            name="BHHH matrix",
            inputs=self.inputs,
            outputs=bhhh(self.ll, self.betas),
        )

    def model_gnorm(self):
        """Loads the function to ``self.gradient_norm()`` to calculate the gradient
        norm of the model cost function.
        """
        self.gradient_norm = function(
            name="Gradient norm",
            inputs=self.inputs,
            outputs=gnorm(self.cost, self.betas),
        )

    def predict(self, db, return_choices=True, split_type="valid"):
        """Returns the predicted choice probabilites"""
        predict_data = db.pandas.inputs(self.inputs, split_type=split_type)
        if not return_choices:
            return self.choice_probabilities(*predict_data)
        else:
            return self.choice_predictions(*predict_data)

    def train(self, db, **kwargs):
        """Function to train the model

        Args:
            db (pycmtensor.Data): database used to train the model
            **kwargs: keyword arguments for adjusting training configuration.
                Possible values are `max_steps:int`, `patience:int`,
                `lr_scheduler:scheduler.Scheduler`, `batch_size:int`. For more
                information and other possible options, see
                `config.Config.hyperparameters`
        """
        self.config.set_hyperparameter(**kwargs)

        # [train-start]
        lr_scheduler = self.config["lr_scheduler"]
        batch_size = self.config["batch_size"]
        max_steps = self.config["max_steps"]
        patience = self.config["patience"]
        patience_increase = self.config["patience_increase"]
        validation_threshold = self.config["validation_threshold"]

        db.n_train_batches = db.n_train_samples // batch_size
        validation_frequency = db.n_train_batches
        max_iterations = max_steps * db.n_train_batches

        # initalize results array
        self.results.performance_graph = OrderedDict()

        # compute the inital results of the model
        self.results.init_loglikelihood = self.loglikelihood(*db.train_data)
        self.results.best_loglikelihood = self.results.init_loglikelihood
        self.results.best_valid_error = self.prediction_error(*db.valid_data)

        # loop parameters
        done_looping = False
        step = 0
        iteration = 0
        shift = 0

        # set learning rate
        learning_rate = lr_scheduler(step)

        # main loop
        start_time = perf_counter()
        info(f"Start (n={db.n_train_samples})")

        while (step < max_steps) and (not done_looping):

            # loop over batch
            learning_rate = lr_scheduler(step)
            for index in range(db.n_train_batches):
                if patience <= iteration:
                    done_looping = True
                    debug(f"Early stopping... (step={step})")
                    break

                # increment iteration
                iteration += 1

                # set index and shift slices
                if self.config["batch_shuffle"]:
                    index = rng.integers(0, db.n_train_batches)
                    shift = rng.integers(0, batch_size)

                # get data slice from dataset
                batch_data = db.get_train_data(self.inputs, index, batch_size, shift)

                # model update step
                self.update_wrt_cost(*batch_data, learning_rate)

                # model validate step
                if iteration % validation_frequency != 0:
                    continue

                train_ll = self.loglikelihood(*db.train_data)
                valid_error = self.prediction_error(*db.valid_data)
                self.results.performance_graph[step] = (
                    np.round(train_ll, 2),
                    np.round(valid_error, 4),
                )

                if valid_error >= self.results.best_valid_error:
                    continue
                msg = f"Best validation error = {valid_error*100:.3f}%, (s={step}, i={iteration}, p={patience}, ll={train_ll:.2f})"
                debug(msg)

                self.results.best_step = step
                self.results.best_iteration = iteration
                self.results.best_loglikelihood = train_ll
                self.results.best_valid_error = valid_error

                self.results.betas = self.betas
                self.results.weights = self.weights

                # increase patience if past validation threshold
                if train_ll > (self.results.best_loglikelihood / validation_threshold):
                    continue

                patience = min(
                    max(patience, iteration * patience_increase), max_iterations
                )

            # increment step
            step += 1

        train_time = round(perf_counter() - start_time, 3)
        self.results.train_time = time_format(train_time)
        self.results.iterations_per_sec = round(iteration / train_time, 2)
        msg = f"End (t={self.results.train_time}, VE={self.results.best_valid_error*100:.2f}%, LL={self.results.best_loglikelihood:.2f}, Step={self.results.best_step})"
        info(msg)

        self.betas = self.results.betas
        self.weights = self.results.weights
        self.results.n_train_samples = db.n_train_samples
        self.results.n_valid_samples = db.n_valid_samples
        self.results.n_params = self.n_params
        self.results.seed = self.config["seed"]
        self.results.lr_history_graph = self.config["lr_scheduler"].history

        # statistical analysis step
        self.results.gnorm = self.gradient_norm(*db.train_data)
        self.results.hessian_matrix = self.H(*db.train_data)
        self.results.bhhh_matrix = self.BHHH(*db.train_data)

    def export_to_pickle(self, f):
        model = self
        pickle.dump(model, f)

    def __str__(self):
        return f"{self.name}"
