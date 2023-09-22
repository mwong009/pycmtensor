from collections import OrderedDict
from time import perf_counter

import aesara.tensor as aet
import numpy as np
from aesara import function, pprint

import pycmtensor.defaultconfig as defaultconfig
from pycmtensor.expressions import Beta, ExpressionParser, Param
from pycmtensor.logger import debug, info, warning
from pycmtensor.models.layers import Layer
from pycmtensor.results import Results
from pycmtensor.utils import time_format

config = defaultconfig.config


class BaseModel(object):
    def __init__(self, **kwargs):
        """Basic model class object

        Attributes:
            name (str): name of the model
            config (pycmtensor.Config): pycmtensor config object
            rng (numpy.random.Generator): random number generator
            params (list): list of model parameters (`betas` & `weights`)
            betas (list): list of model scalar betas
            sigmas (list): list of model scalar sigmas
            weights (list): list of model weight matrices
            biases (list): list of model vector biases
            updates (list): list of (param, update) tuples
            learning_rate (TensorVariable): symbolic reference to the learning rate
            results (Results): stores the results of the model estimation
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.results = Results()

        for key, value in kwargs.items():
            self.config.add(key, value)

        debug(f"Building model...")

    @property
    def n_params(self):
        return self.n_betas + self.n_weights + self.n_biases

    @property
    def n_betas(self):
        if "betas" in dir(self):
            return len(self.betas)
        return 0

    @property
    def n_weights(self):
        if "weights" in dir(self):
            return np.sum([np.prod(w.shape) for w in self.weights])
        return 0

    @property
    def n_biases(self):
        if "biases" in dir(self):
            return np.sum([np.prod(b.shape) for b in self.biases])
        return 0

    def get_weights(self):
        """returns the values of the weights

        Returns:
            (dict): weight values
        """
        if "weights" in dir(self):
            return {w.name: w.get_value() for w in self.weights}
        return {}

    def get_biases(self):
        """returns the values of the biases

        Returns:
            (dict): biases values
        """
        if "biases" in dir(self):
            return {b.name: b.get_value() for b in self.biases}
        return {}

    def get_betas(self):
        """returns the values of the betas

        Returns:
            (dict): beta values
        """
        if "betas" in dir(self):
            return {beta.name: beta.get_value() for beta in self.betas}
        return {}

    def reset_values(self):
        """resets the values of all parameters"""
        if "params" in dir(self):
            for p in self.params:
                p.reset_value()

    def include_params_for_convergence(self, *args, **kwargs):
        """dummy method for additional parameter objects for calculating convergence

        Args:
            *args (None): overloaded arguments
            **kwargs (dict): overloaded keyword arguments

        Returns:
            (OrderedDict): a dictonary of addition parameters to include
        """
        return OrderedDict()

    def include_regularization_terms(self, *regularizers):
        """dummy method for additional regularizers into the cost function

        Args:
            *args (None): overloaded arguments

        Returns:
            (list[TensorVariable]): a list of symbolic variables that specify additional regualrizers to minimize against
        """

        if len(regularizers) > 0:
            for reg in regularizers:
                self.cost += reg

    def build_cost_updates_fn(self, updates):
        self.cost_updates_fn = function(
            name="cost_updates",
            inputs=self.x + [self.y, self.learning_rate, self.index],
            outputs=self.cost,
            updates=updates,
            allow_input_downcast=True,
        )

    def predict(self, ds, return_probabilities=False):
        if not "choice_probabilities_fn" in dir(self):
            self.choice_probabilities_fn = function(
                name="choice_probabilities",
                inputs=self.x,
                outputs=self.p_y_given_x,
                allow_input_downcast=True,
            )

        valid_data = ds.valid_dataset(self.x)

        prob = self.choice_probabilities_fn(*valid_data)

        if return_probabilities:
            return {i: prob[i] for i in range(prob.shape[0])}
        else:
            return {"pred_" + ds.choice: np.argmax(prob, axis=0)}

    def elasticities(self, ds, wrt_choice):
        p_y_given_x = self.p_y_given_x[self.y, ..., self.index]
        while p_y_given_x.ndim > 1:
            p_y_given_x = aet.sum(p_y_given_x, axis=1)
        dy_dx = aet.grad(aet.sum(p_y_given_x), self.x, disconnected_inputs="ignore")

        if not "elasticity_fn" in dir(self):
            self.elasticity_fn = function(
                inputs=self.x + [self.y, self.index],
                outputs={x.name: g * x / p_y_given_x for g, x in zip(dy_dx, self.x)},
                on_unused_input="ignore",
                allow_input_downcast=True,
            )
        train_data = ds.train_dataset(self.x)
        index = np.arange((len(train_data[-1])))
        choice = (np.ones(shape=index.shape) * wrt_choice).astype(int)
        return self.elasticity_fn(*train_data, choice, index)

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return pprint(self.cost)

    def __getattr__(self, name):
        if (name == "hessian_fn") or (name == "gradient_vector_fn"):
            self.build_gh_fn()
            return getattr(self, name)
        else:
            return False

    @staticmethod
    def extract_params(cost, variables):
        """Extracts Param objects from variables

        Args:
            cost (TensorVariable): function to evaluate
            variables (Union[dict, list]): list of variables from the current program
        """
        params = []
        symbols = ExpressionParser.parse(cost)
        seen = set()

        if isinstance(variables, dict):
            variables = [v for _, v in variables.items()]

        for variable in variables:
            if (not isinstance(variable, Param)) or isinstance(variable, Layer):
                continue

            if isinstance(variable, Param) and (variable.name in seen):
                continue

            if variable.name not in symbols:
                # raise a warning if variable is not in any utility function
                warning(f"{variable.name} not in any utility functions")
                continue

            params.append(variable)
            seen.add(variable.name)

        return params

    @staticmethod
    def drop_unused_variables(cost, params, variables):
        """Internal method to remove ununsed tensors

        Args:
            cost (TensorVariable): function to evaluate
            params (Param): param objects
            variables (dict): list of array variables from the dataset

        Returns:
            (list): a list of param names which are not used in the model
        """
        symbols = ExpressionParser.parse(cost)
        param_names = [p.name for p in params]
        symbols = [s for s in symbols if s not in param_names]
        return [var for var in list(variables) if var not in symbols]


def compute(model, ds, **params):
    """Function for manual computation of model by specifying parameters as arguments

    Args:
        model (pycmtensor.models.BaseModel): model to train
        ds (pycmtensor.dataset.Dataset): dataset to use for training
        **params (dict): keyword arguments for model coefficients (`Params`)

    Returns:
        dict: model likelihood and error for training and validation datasets

    Example:
    ```
    compute(model, ds, b_time=-5.009, b_purpose=0.307, asc_pt=-1.398, asc_drive=4.178,
            asc_cycle=-3.996, b_car_own=1.4034)
    ```
    """
    # saves original values and replace values by test values in params
    p_value_old = {}
    for p in model.params:
        if p.name in params:
            p_value_old[p.name] = p.get_value()
            p.set_value(params[p.name])

    # compute all the outputs of the training and validation datasets
    x_y = model.x + [model.y]
    train_data = ds.train_dataset(x_y)
    valid_data = ds.valid_dataset(x_y)

    t_index = np.arange(len(train_data[-1]))
    v_index = np.arange(len(valid_data[-1]))

    t_log_likelihood = model.log_likelihood_fn(*train_data, t_index)
    t_error = model.prediction_error_fn(*train_data)

    v_log_likelihood = model.log_likelihood_fn(*valid_data, v_index)
    v_error = model.prediction_error_fn(*valid_data)

    # put back original values
    for p in model.params:
        if p.name in p_value_old:
            p.set_value(p_value_old[p.name])

    # output results
    return {
        "train log likelihood": t_log_likelihood,
        "train error": t_error,
        "validation log likelihood": v_log_likelihood,
        "validation error": v_error,
    }


def train(model, ds, **kwargs):
    """main training loop

    Args:
        model (pycmtensor.models.BaseModel): model to train
        ds (pycmtensor.dataset.Dataset): dataset to use for training
        **kwargs (dict): overloaded keyword arguments. See [configuration](../../../user_guide/configuration.md) in the user guide for details on possible options
    """
    for key, value in kwargs.items():
        model.config.add(key, value)

    batch_size = model.config.batch_size
    if (batch_size == 0) or (batch_size == None):
        model.config.batch_size = ds.n_train
        batch_size = model.config.batch_size

    n_train = ds.n_train
    n_valid = ds.n_valid
    n_train_batches = n_train // batch_size
    model.config.BFGS_warmup = model.config.BFGS_warmup * n_train_batches

    patience = max(model.config.patience, n_train_batches)
    patience_inc = model.config.patience_increase
    validation_threshold = model.config.validation_threshold
    convergence_threshold = model.config.convergence_threshold
    validation_freq = n_train_batches
    max_epochs = model.config.max_epochs
    max_iterations = max_epochs * n_train_batches
    acceptance_method = model.config.acceptance_method

    done_looping = False
    converged = False
    epoch = 0
    iteration = 0
    shift = 0
    gnorm_min = np.inf

    optimizer = model.config.optimizer(model.params, config=model.config)
    updates = optimizer.update(model.cost, model.params, model.learning_rate)
    model.build_cost_updates_fn(updates)

    lr_scheduler = model.config.lr_scheduler(
        lr=model.config.base_learning_rate,
        factor=model.config.lr_stepLR_factor,
        drop_every=model.config.lr_stepLR_drop_every,
        power=model.config.lr_PolynomialLR_power,
        cycle_steps=model.config.lr_CLR_cycle_steps,
        gamma=model.config.lr_ExpRangeCLR_gamma,
    )

    loglikelihood_graph = []
    error_graph = []

    x_y = model.x + [model.y]
    train_data = ds.train_dataset(x_y)
    valid_data = ds.valid_dataset(x_y)

    t_index = np.arange(len(train_data[-1]))
    log_likelihood = model.log_likelihood_fn(*train_data, t_index)
    error = model.prediction_error_fn(*valid_data)

    model.results.null_loglikelihood = log_likelihood
    model.results.best_loglikelihood = log_likelihood
    model.results.best_valid_error = error
    model.results.best_epoch = 0
    model.results.gnorm = np.nan
    model.results.n_train = n_train
    model.results.n_valid = n_valid
    model.results.n_params = model.n_params
    model.results.seed = model.config.seed

    loglikelihood_graph.append((0, log_likelihood))
    error_graph.append((0, error))

    params_prev = [p.get_value() for p in model.params if isinstance(p, Beta)]

    model.results.betas = OrderedDict(
        {p.name: p.get_value() for p in model.params if isinstance(p, Beta)}
    )
    model.results.params = OrderedDict({p.name: p.get_value() for p in model.params})

    p = model.include_params_for_convergence(train_data, t_index)
    params_prev.extend(list(p.values()))
    model.results.betas.update(p)

    batch_data = []
    for batch in range(n_train_batches):
        batch_data.append(ds.train_dataset(x_y, batch, batch_size, shift))

    start_time = perf_counter()
    info(
        f"Start (n={n_train}, epoch={epoch}, LL={model.results.null_loglikelihood:.2f}, error={model.results.best_valid_error*100:.2f}%)"
    )

    while (epoch < max_epochs) and (not done_looping):
        learning_rate = lr_scheduler(epoch)  # get learning rate at this epoch

        for i in range(n_train_batches):
            index = np.arange(len(batch_data[i][-1]))
            model.cost_updates_fn(*batch_data[i], learning_rate, index)  # update model

            if iteration % validation_freq == 0:
                # training loglikelihood
                log_likelihood = model.log_likelihood_fn(*train_data, t_index)
                loglikelihood_graph.append((iteration, log_likelihood))

                # validation error
                error = model.prediction_error_fn(*valid_data)
                error_graph.append((iteration, error))

                # convergence
                params = [p.get_value() for p in model.params if isinstance(p, Beta)]
                p = model.include_params_for_convergence(train_data, t_index)
                params.extend(list(p.values()))
                diff = [p_prev - p for p_prev, p in zip(params_prev, params)]
                params_prev = params
                gnorm = np.sqrt(np.sum(np.square(diff)))

                # stdout
                best_ll = model.results.best_loglikelihood
                best_err = model.results.best_valid_error
                vt = validation_threshold

                if acceptance_method == 1:
                    accept = log_likelihood > best_ll
                else:
                    accept = error < best_err

                if (
                    (gnorm < (gnorm_min / 5.0))
                    or ((epoch % (max_epochs // 10)) == 0)
                    or (accept and (log_likelihood > (best_ll / (vt * vt))))
                    or ((1 - accept) and (error < (best_err / (vt * vt))))
                ):
                    if gnorm < (gnorm_min / 5.0):
                        gnorm_min = gnorm
                    info(
                        f"Train (epoch={epoch}, LL={log_likelihood:.2f}, error={error*100:.2f}%, gnorm={gnorm:.5e}, {iteration}/{patience})"
                    )

                # acceptance of new results
                if accept:
                    if log_likelihood > (best_ll / validation_threshold):
                        patience = int(
                            min(max(patience, iteration * patience_inc), max_iterations)
                        )

                    # save best results if new estimated model is accepted
                    model.results.best_epoch = epoch
                    model.results.best_iteration = iteration
                    model.results.best_loglikelihood = log_likelihood
                    model.results.best_valid_error = error
                    model.results.gnorm = gnorm

                    # save Beta params
                    for p in model.params:
                        model.results.params[p.name] = p.get_value()
                        if isinstance(p, Beta):
                            model.results.betas[p.name] = p.get_value()
                    if "tn_betas_fn" in dir(model):
                        for key, value in model.tn_betas_fn(*train_data, index).items():
                            model.results.betas[key] = value

                # set condition for convergence
                if gnorm < convergence_threshold:
                    converged = True
                else:
                    patience = int(
                        min(max(patience, iteration * patience_inc), max_iterations)
                    )

            iteration += 1

            # secondary condition for convergence
            if (iteration > patience) or converged:
                iteration -= 1
                info(
                    f"Train (epoch={epoch}, LL={log_likelihood:.2f}, error={error*100:.2f}%, gnorm={gnorm:.5e}, {iteration}/{patience})"
                )

                done_looping = True  # break loop if convergence reached
                break

        epoch += 1  # increment epoch

    train_time = round(perf_counter() - start_time, 3)
    model.results.train_time = time_format(train_time)
    model.results.epochs_per_sec = round(epoch / train_time, 2)
    if converged:
        info(f"Model converged (t={train_time})")
    else:
        info(
            f"Maximum number of epochs reached: {iteration}/{patience} (t={train_time})"
        )

    model.results.lr_history_graph = lr_scheduler.history
    model.results.loglikelihood_graph = loglikelihood_graph
    model.results.error_graph = error_graph

    debug(f"Evaluating full hessian and bhhh matrix")
    for p in model.params:
        p.set_value(model.results.params[p.name])

    n_betas = len(model.results.betas)
    gradient_vector = np.zeros((n_train, n_betas))
    for n in range(n_train):
        data = [[d[n]] for d in train_data]
        gradient_vector[n, :] = model.gradient_vector_fn(*data, np.array([0]))
    bhhh = np.sum(gradient_vector[:, :, None] * gradient_vector[:, None, :], axis=0)

    index = np.arange(len(train_data[-1]))
    hessian = model.hessian_fn(*train_data, index)

    model.results.bhhh_matrix = bhhh
    model.results.hessian_matrix = hessian

    info(
        f"Best results obtained at epoch {model.results.best_epoch}: LL={model.results.best_loglikelihood:.2f}, error={model.results.best_valid_error*100:.2f}%, gnorm={model.results.gnorm:.5e}"
    )
