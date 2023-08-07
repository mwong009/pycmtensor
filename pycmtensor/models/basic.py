from collections import OrderedDict
from time import perf_counter

import numpy as np

import pycmtensor.defaultconfig as defaultconfig
from pycmtensor.expressions import Beta, ExpressionParser, Param
from pycmtensor.logger import debug, info, warning
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
        self.weights = []
        self.biases = []
        self.betas = []
        self.results = Results()

        for key, value in kwargs.items():
            self.config.add(key, value)

        debug(f"Building model...")

    @property
    def n_params(self):
        """Return the total number of estimated parameters"""
        return self.n_betas + self.n_weights + self.n_biases

    @property
    def n_betas(self):
        """Return the number of estimated Betas"""
        return len(self.betas)

    @property
    def n_weights(self):
        """Return the total number of estimated Weight parameters"""
        return np.sum([np.prod(w.shape) for w in self.weights])

    @property
    def n_biases(self):
        """Return the total number of estimated Weight parameters"""
        return np.sum([np.prod(b.shape) for b in self.biases])

    def get_weights(self):
        """Returns a dict of Weight values"""
        return {w.name: w.get_value() for w in self.weights}

    def get_biases(self):
        """Returns a dict of Weight values"""
        return {b.name: b.get_value() for b in self.biases}

    def get_betas(self):
        """Returns a dict of Beta values"""
        return {beta.name: beta.get_value() for beta in self.betas}

    def reset_values(self):
        """Resets Model parameters to their initial value"""
        for p in self.params:
            p.reset_value()

    def include_params_for_convergence(self, **kwargs):
        """Returns a Ordered dict of parameters values to check for convergence

        Returns:
            (OrderedDict): ordered dictionary of parameter values
        """
        return OrderedDict()


def extract_params(cost, variables):
    """Extracts Param objects from variables

    Args:
        cost (TensorVariable): function to evaluate
        variables (Union[dict, list]): list of variables from the current program
    """
    params = []
    symbols = ExpressionParser().parse(cost)
    seen = set()

    if isinstance(variables, dict):
        variables = [v for _, v in variables.items()]

    for variable in variables:
        if (not isinstance(variable, Param)) or (variable.name in seen):
            continue

        if variable.name not in symbols:
            # raise a warning if variable is not in any utility function
            warning(f"{variable.name} not in any utility functions")
            continue

        params.append(variable)
        seen.add(variable.name)

    return params


def drop_unused_variables(cost, params, variables):
    """Internal method to remove ununsed tensors

    Args:
        cost (TensorVariable): function to evaluate
        params (Param): param objects
        variables (dict): list of array variables from the dataset

    Returns:
        (list): a list of param names which are not used in the model
    """
    symbols = ExpressionParser().parse(cost)
    param_names = [p.name for p in params]
    symbols = [s for s in symbols if s not in param_names]
    return [var for var in list(variables) if var not in symbols]


def train(model, ds, **kwargs):
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
    max_steps = model.config.max_steps
    max_iterations = max_steps * n_train_batches

    done_looping = False
    converged = False
    step = 0
    iteration = 0
    shift = 0
    gnorm_tol = np.inf

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

    performance_graph = OrderedDict()

    x_y = model.x + [model.y]
    train_data = ds.train_dataset(x_y)
    valid_data = ds.valid_dataset(x_y)

    t_index = np.arange(len(train_data[-1]))
    log_likelihood = model.log_likelihood_fn(*train_data, t_index)
    error = model.prediction_error_fn(*valid_data)

    model.results.null_loglikelihood = log_likelihood
    model.results.best_loglikelihood = log_likelihood
    model.results.best_valid_error = error
    model.results.best_step = 0
    model.results.gnorm = np.nan
    model.results.n_train = n_train
    model.results.n_valid = n_valid
    model.results.n_params = model.n_params
    model.results.seed = model.config.seed

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
        f"Start (n={n_train}, Step={step}, LL={model.results.null_loglikelihood:.2f}, Error={model.results.best_valid_error*100:.2f}%)"
    )

    while (step < max_steps) and (not done_looping):
        learning_rate = lr_scheduler(step)  # get learning rate at this step

        for i in range(n_train_batches):
            index = np.arange(len(batch_data[i][-1]))
            model.cost_updates_fn(*batch_data[i], learning_rate, index)  # update model

            if iteration % validation_freq == 0:
                log_likelihood = model.log_likelihood_fn(*train_data, t_index)
                performance_graph[step] = {"log likelihood": log_likelihood}

                params = [p.get_value() for p in model.params if isinstance(p, Beta)]
                p = model.include_params_for_convergence(train_data, t_index)
                params.extend(list(p.values()))

                diff = [p_prev - p for p_prev, p in zip(params_prev, params)]
                params_prev = params

                gnorm = np.sqrt(np.sum(np.square(diff)))

                if gnorm < (gnorm_tol / 5.0):
                    gnorm_tol = gnorm
                    info(
                        f"Train (Step={step}, LL={log_likelihood:.2f}, Error={error*100:.2f}%, gnorm={gnorm:.5e}, {iteration}/{patience})"
                    )

                if log_likelihood > model.results.best_loglikelihood:
                    # validate if loglikelihood improves
                    error = model.prediction_error_fn(*valid_data)

                    this_ll = log_likelihood
                    best_ll = model.results.best_loglikelihood
                    if this_ll > best_ll / validation_threshold:
                        # increase patience if model is not converged
                        patience = int(
                            min(max(patience, iteration * patience_inc), max_iterations)
                        )

                    model.results.best_step = step
                    model.results.best_iteration = iteration
                    model.results.best_loglikelihood = log_likelihood
                    model.results.best_valid_error = error
                    model.results.gnorm = gnorm

                    for p in model.params:
                        model.results.params[p.name] = p.get_value()
                        if isinstance(p, Beta):
                            model.results.betas[p.name] = p.get_value()

                    p = model.include_params_for_convergence(train_data, t_index)
                    model.results.betas.update(p)

                if gnorm < convergence_threshold:
                    converged = True
                else:
                    patience = int(
                        min(max(patience, iteration * patience_inc), max_iterations)
                    )

            iteration += 1

            # reached convergence
            if (iteration > patience) or converged:
                iteration -= 1
                info(
                    f"Train (Step={step}, LL={log_likelihood:.2f}, Error={error*100:.2f}%, gnorm={gnorm:.5e}, {iteration}/{patience})"
                )

                done_looping = True  # break loop if convergence reached
                break

        step += 1  # increment step

    train_time = round(perf_counter() - start_time, 3)
    model.results.train_time = time_format(train_time)
    model.results.iterations_per_sec = round(iteration / train_time, 2)
    if converged:
        info(f"Model converged (t={train_time})")
    else:
        info(
            f"Maximum number of iterations reached: {iteration}/{patience} (t={train_time})"
        )

    model.results.lr_history_graph = lr_scheduler.history
    model.results.performance_graph = performance_graph

    for p in model.params:
        p.set_value(model.results.params[p.name])

    n_betas = len(model.results.betas)
    gradient_vector = np.zeros((n_train, n_betas))
    hessian = np.zeros((n_train, n_betas, n_betas))
    for n in range(n_train):
        data = [[d[n]] for d in train_data]
        gradient_vector[n, :] = model.gradient_vector_fn(*data, np.array([0]))
        hessian[n, :, :] = model.hessian_fn(*data, np.array([0]))

    bhhh = gradient_vector[:, :, None] * gradient_vector[:, None, :]

    model.results.bhhh_matrix = bhhh
    model.results.hessian_matrix = hessian

    info(
        f"Best results obtained at Step {model.results.best_step}: LL={model.results.best_loglikelihood:.2f}, Error={model.results.best_valid_error*100:.2f}%, gnorm={model.results.gnorm:.5e}"
    )
