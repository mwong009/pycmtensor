# pycmtensor.py
"""PyCMTensor main module"""
import timeit
from collections import OrderedDict

import dill as pickle
import numpy as np

from pycmtensor import config, rng

from .expressions import Beta, ExpressionParser, Weights
from .logger import log
from .results import Results
from .utils import time_format


class PyCMTensorModel:
    """Base model class object"""

    def __init__(self, db):
        self.name = "PyCMTensorModel"
        self.config = config
        self.params = []  # keep track of all params
        self.betas = []  # keep track of the Betas
        self.weights = []  # keep track of the Weights
        self.inputs = db.all
        self.results = Results()

    def add_params(self, params):
        """Method to load locally defined variables

        Args:
                params (dict or list): a dict or list of items which are :class:`expressions.Beta` or :class:`expressions.Weights`
        """
        if not isinstance(params, (dict, list)):
            msg = "params must be of Type dict or list"
            log(40, msg)
            raise TypeError(msg)

        # create a dict of str(param): param, if params is given as a list
        if isinstance(params, list):
            params = {str(p): p for p in params}

        # iterate through the dict of params:
        if not hasattr(self, "cost"):
            log(40, "No valid cost function defined.")
            raise ReferenceError

        symbols = ExpressionParser().parse(getattr(self, "cost"))
        seen = set()
        for _, p in params.items():
            if not isinstance(p, (Beta, Weights)):
                continue

            if p.name in seen:
                msg = f"Duplicate param names defined: {p.name}."
                log(40, msg)
                raise NameError(msg)

            if (p.name not in symbols) and (p.status == 0):
                msg = f"Unused Beta {p.name} removed from computational graph"
                log(30, msg)
                continue

            self.params.append(p)
            seen.add(p.name)

            if isinstance(p, (Beta)):
                self.betas.append(p)
            else:
                self.weights.append(p)

        self.n_params = self.get_n_params()

    def add_regularizers(self, l_reg):
        """Adds regularizer ``l_reg`` to model cost function"""
        if not hasattr(self, "cost"):
            log(40, "No valid cost function defined.")
            raise ReferenceError

        self.cost += l_reg

    def get_n_params(self):
        """Get the total number of parameters"""
        return self.get_n_betas() + sum(self.get_n_weights())

    def get_n_betas(self):
        """Get the count of Beta parameters"""
        return len(self.betas)

    def get_betas(self) -> dict:
        """Returns the Beta (key, value) pairs as a dict"""
        return {beta.name: beta.get_value() for beta in self.betas}

    def get_n_weights(self):
        """Get the total size of each weight matrix"""
        return [w.size for w in self.get_weights()]

    def get_weights(self) -> list[np.ndarray]:
        """Returns the Weights as a list of matrices"""
        return [w.get_value() for w in self.weights]

    def reset_values(self):
        """Resets param values to their initial values"""
        for p in self.params:
            p.reset_value()

    def predict(self, db, return_choices=True, split_type=None):
        """Returns the predicted choice probabilites"""
        predict_data = db.pandas.inputs(self.inputs, split_type=split_type)
        if not return_choices:
            return self.choice_probabilities(*predict_data)
        else:
            return self.choice_predictions(*predict_data)

    def train(self, db, steps=np.inf, k=0):
        """Function to train the model"""
        self.config.check_values()  # assert config.hyperparameters

        # [train-start]
        lr_scheduler = self.config["lr_scheduler"]
        batch_size = self.config["batch_size"]
        max_steps = min(self.config["max_steps"], steps)
        patience = self.config["patience"]
        patience_increase = self.config["patience_increase"]
        validation_threshold = self.config["validation_threshold"]

        if db.split_frac is not None:
            n_train_samples = len(db.pandas.train_dataset[k])
            n_valid_samples = len(db.pandas.valid_dataset[k])
        else:
            n_train_samples = db.get_nrows()
            n_valid_samples = db.get_nrows()
        n_train_batches = n_train_samples // batch_size
        validation_frequency = n_train_batches
        max_iterations = max_steps * n_train_batches
        msg = f"batch_size={batch_size}, max_steps={max_steps}, n_samples={n_train_samples}"
        log(10, msg)

        # initalize results array
        self.results.performance_graph = OrderedDict()

        # set training and validation datasets
        train_data = db.pandas.inputs(self.inputs, split_type="train")
        valid_data = db.pandas.inputs(self.inputs, split_type="valid")

        # compute the inital results of the model
        self.results.init_loglikelihood = self.loglikelihood(*train_data)
        self.results.best_loglikelihood = self.results.init_loglikelihood
        self.results.best_valid_error = self.prediction_error(*valid_data)

        # loop parameters
        done_looping = False
        step = 0
        iteration = 0
        shift = 0

        # set learning rate
        learning_rate = lr_scheduler(step)

        # main loop
        start_time = timeit.default_timer()
        log(20, f"Start (n={n_train_samples})")

        while (step < max_steps) and (not done_looping):

            # loop over batch
            learning_rate = lr_scheduler(step)
            for index in range(n_train_batches):
                if patience <= iteration:
                    done_looping = True
                    log(10, f"Early stopping... (s={step})")
                    break

                # increment iteration
                iteration += 1

                # set index and shift slices
                if self.config["batch_shuffle"]:
                    index = rng.integers(0, n_train_batches)
                    shift = rng.integers(0, batch_size)

                # get data slice from dataset
                batch_data = db.pandas.inputs(
                    self.inputs, index, batch_size, shift, split_type="train"
                )

                # model update step
                self.update_wrt_cost(*batch_data, learning_rate)

                # model validate step
                if iteration % validation_frequency != 0:
                    continue

                train_ll = self.loglikelihood(*train_data)
                valid_error = self.prediction_error(*valid_data)
                self.results.performance_graph[step] = (
                    np.round(train_ll, 2),
                    np.round(valid_error, 4),
                )

                if valid_error >= self.results.best_valid_error:
                    continue

                msg = f"Best validation error = {valid_error*100:.3f}%, (s={step}, i={iteration}, p={patience}, ll={train_ll:.2f})"
                log(10, msg)

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

        train_time = round(timeit.default_timer() - start_time, 3)
        self.results.train_time = time_format(train_time)
        self.results.iterations_per_sec = round(iteration / train_time, 2)
        msg = f"End (t={self.results.train_time}, VE={self.results.best_valid_error*100:.3f}%, LL={self.results.best_loglikelihood}, S={self.results.best_step})"
        log(20, msg)

        self.betas = self.results.betas
        self.weights = self.results.weights
        self.results.n_train_samples = n_train_samples
        self.results.n_valid_samples = n_valid_samples
        self.results.n_params = self.n_params
        self.results.seed = self.config["seed"]
        self.results.lr_history_graph = self.config["lr_scheduler"].history

        # statistical analysis step
        self.results.gnorm = self.gradient_norm(*train_data)
        self.results.hessian_matrix = self.H(*train_data)
        self.results.bhhh_matrix = self.BHHH(*train_data)

    def export_to_pickle(self, f):
        model = self
        pickle.dump(model, f)

    def __str__(self):
        return f"{self.name}"
