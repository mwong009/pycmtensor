from collections import OrderedDict
from time import perf_counter

import numpy as np
import pandas as pd

from pycmtensor.expressions import Beta
from pycmtensor.functions import f1_score
from pycmtensor.logger import debug, info, warning
from pycmtensor.utils import human_format as hf
from pycmtensor.utils import time_format

IS_TRAINING = 1


def compute(model, ds, lr_scheduler, update=False, **params):
    """Function for manual computation of model by specifying parameters as arguments

    Args:
        model (pycmtensor.models.BaseModel): model to train
        ds (pycmtensor.dataset.Dataset): dataset to use for training
        update (bool): if True, run `cost_updates_fn` once before computing the model
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
    p_value_old = {p.name: p.get_value() for p in model.params if p.name in params}
    for p in model.params:
        if p.name in params:
            p.set_value(params[p.name])

    # compute all the outputs of the training and validation datasets
    x_y = model.x + [model.y]
    train_data = ds.train_dataset(x_y)
    valid_data = ds.valid_dataset(x_y)

    t_index = np.arange(len(train_data[-1]))
    v_index = np.arange(len(valid_data[-1]))

    if update:
        model.lr_scheduler = lr_scheduler
        learning_rate = model.lr_scheduler(0)
        model.cost_updates_fn(*train_data, learning_rate, t_index, IS_TRAINING)

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


def train(model, ds, optimizer, lr_scheduler, **kwargs):
    """Main training loop

    Args:
        model (pycmtensor.models.BaseModel): model to train
        ds (pycmtensor.dataset.Dataset): dataset to use for training
        **kwargs (dict): overloaded keyword arguments. See [configuration](../../../user_guide/configuration.md) in the user guide for details on possible options
    """
    for key, value in kwargs.items():
        model.config.add(key, value)

    model.batch_size = kwargs.get("batch_size", None)
    model.patience = kwargs.get("patience", 2000)
    model.patience_increase = kwargs.get("patience_increase", 2)
    model.convergence_threshold = kwargs.get("convergence_threshold", 1e-4)

    model.n_train = ds.n_train
    model.n_valid = ds.n_valid

    if (model.batch_size == 0) or (model.batch_size == None):
        model.batch_size = model.n_train

    model.n_train_batches = model.n_train // model.batch_size
    model.validation_freq = model.n_train_batches

    model.optimizer = optimizer
    updates = model.optimizer.update(model.cost, model.params, model.learning_rate)
    model.build_cost_updates_fn(updates)

    model.lr_scheduler = lr_scheduler
    model.max_epochs = model.lr_scheduler.max_epochs
    model.max_iterations = model.max_epochs * model.n_train_batches

    model.stats = pd.DataFrame(
        columns=[
            "train_log_likelihood",
            "train_error",
            "valid_log_likelihood",
            "valid_error",
            "gnorm",
            "learning_rate",
        ]
    )

    x_y = model.x + [model.y]
    train_data = ds.train_dataset(x_y)
    valid_data = ds.valid_dataset(x_y)

    t_index = np.arange(len(train_data[-1]))
    log_like = model.log_likelihood_fn(*train_data, t_index)
    null_loglikelihood = model.null_log_likelihood_fn(*train_data, t_index)
    train_error = model.prediction_error_fn(*train_data)

    if set(ds.idx_train) != set(ds.idx_valid):
        valid_error = model.prediction_error_fn(*valid_data)
    else:
        valid_error = train_error

    model.results.best_loglikelihood = log_like
    model.results.best_valid_error = valid_error
    model.results.best_train_error = train_error
    model.results.best_epoch = 0
    model.results.gnorm = np.nan
    model.results.gnorm_min = np.inf
    model.results.init_loglikelihood = log_like
    model.results.null_loglikelihood = null_loglikelihood

    done = False
    epoch = 0
    i = 0

    model.results.statistics_graph = {
        "train_ll": [],
        "train_error": [],
        "valid_error": [],
    }

    model.results.n_train = model.n_train
    model.results.n_valid = model.n_valid
    model.results.n_params = model.n_params
    model.results.config = model.config
    model.results.converged = [False, "0"]

    params_prev = initialize_results(model, train_data, t_index)
    batch_data = []
    for b in range(model.n_train_batches):
        batch_data.append(ds.train_dataset(x_y, b, model.batch_size, shift=0))

    info(
        f"Start (n={model.n_train}, nll={model.results.null_loglikelihood:.2f}, error={model.results.best_valid_error*100:.2f}%)"
    )

    model.start_time = perf_counter()
    while (epoch < model.max_epochs) and (not done):
        learning_rate = model.lr_scheduler(epoch)  # get learning rate

        for b in range(model.n_train_batches):
            index = np.arange(len(batch_data[b][-1]))
            model.cost_updates_fn(*batch_data[b], learning_rate, index, IS_TRAINING)

            if i % model.validation_freq == 0:
                # train step
                log_like, train_error, valid_error = train_step(
                    model, ds, train_data, t_index, valid_data
                )

                # convergence check
                gnorm, params_prev = gnorm_func(model, train_data, t_index, params_prev)

                append_stats(
                    model, log_like, train_error, valid_error, learning_rate, gnorm, i
                )

                # verbose logging
                verbose_logging(
                    model, gnorm, epoch, i, log_like, valid_error, learning_rate
                )

                # condition for acceptance
                accept = accept_condition(model, i, log_like, valid_error)
                if accept:
                    model.save_beta_params(train_data, index)
                    model.save_statistics(
                        epoch, i, log_like, valid_error, train_error, gnorm
                    )

                # check condition for termination
                model.results.converged = early_stopping(model, gnorm, i)

            i += 1

            # secondary condition for convergence
            if done_looping(
                model, epoch, i, log_like, valid_error, gnorm, learning_rate
            ):
                done = True
                break

        epoch += 1  # increment epoch

    model = post_training(model, epoch)

    # Save hessian data
    model.save_hessian_data(model.n_train, train_data)


def post_training(model, epoch):
    now = perf_counter()
    train_time = round(now - model.start_time, 3)
    model.results.train_time = time_format(train_time)
    model.results.epochs_per_sec = round(epoch / train_time, 2)

    threshold = model.convergence_threshold
    if model.results.converged[1] == "gnorm < threshold":
        info(f"Model converged (t={train_time}), gnorm < {threshold}")
    else:
        info(f"Max iters reached: {hf(i-1)}/{hf(model.patience)} (t={train_time})")

    # save statistics
    for key, value in model.results.statistics_graph.items():
        model.results.statistics_graph[key] = np.array(value).tolist()
    # model.results.statistics_graph["learning_rate"] = model.lr_scheduler.history

    best_epoch = model.results.best_epoch
    best_ll = model.results.best_loglikelihood
    best_error = model.results.best_valid_error
    gnorm = model.results.gnorm

    info(
        f"Best results obtained at epoch {best_epoch}: LL={best_ll:.2f}, error={best_error*100:.2f}%, gnorm={gnorm:.5e}"
    )

    return model


def initialize_results(model, train_data, t_index):
    model.results.betas = OrderedDict(
        {p.name: p.get_value() for p in model.params if isinstance(p, Beta)}
    )
    model.results.params = OrderedDict({p.name: p.get_value() for p in model.params})
    params_prev = [p.get_value() for p in model.params if isinstance(p, Beta)]
    p = model.include_params_for_convergence(train_data, t_index)
    params_prev.extend(list(p.values()))
    model.results.betas.update(p)
    return params_prev


def train_step(model, ds, train_data, t_index, valid_data):
    log_like = model.log_likelihood_fn(*train_data, t_index)
    valid_error = model.prediction_error_fn(*valid_data)
    if set(ds.idx_train) != set(ds.idx_valid):
        train_error = model.prediction_error_fn(*train_data)
    else:
        train_error = valid_error
    return log_like, train_error, valid_error


def append_stats(
    model, log_like, train_error, valid_error, learning_rate, gnorm, iteration
):
    model.results.statistics_graph["train_ll"].append(log_like)
    model.results.statistics_graph["valid_error"].append(valid_error)
    model.results.statistics_graph["train_error"].append(train_error)

    model.stats.loc[iteration, "train_log_likelihood"] = log_like
    model.stats.loc[iteration, "train_error"] = train_error
    model.stats.loc[iteration, "valid_log_likelihood"] = log_like
    model.stats.loc[iteration, "valid_error"] = valid_error
    model.stats.loc[iteration, "gnorm"] = gnorm
    model.stats.loc[iteration, "learning_rate"] = learning_rate


def done_looping(model, epoch, i, log_like, error, gnorm, learning_rate):
    if (i > model.patience) or (model.results.converged[0] != False):
        info(
            f"Train {hf(i-1)}/{hf(model.patience)} (epoch={hf(epoch)}, LL={log_like:.2f}, error={error*100:.2f}%, gnorm={gnorm:.3e}), lr={learning_rate:.2e}"
        )
        return True
    return False


def gnorm_func(model, train_data, t_index, params_prev):
    params = [p.get_value() for p in model.params if isinstance(p, Beta)]
    q = model.include_params_for_convergence(train_data, t_index)
    params.extend(list(q.values()))
    diff = [q_prev - q for q_prev, q in zip(params_prev, params)]
    return np.sqrt(np.sum(np.square(diff))), params


def accept_condition(model, i, ll, error):
    """Determines if the current iteration meets the acceptance criteria based on model configuration.

    Args:
        model (pycmtensor.models.BaseModel): The model being evaluated.
        i (int): The current iteration number.
        ll (float): The log likelihood of the model at the current iteration.
        error (float): The prediction error of the model at the current iteration.

    Returns:
        bool: Whether the current iteration meets the acceptance criteria.
    """
    accept = False
    acceptance_method = model.config.acceptance_method

    if acceptance_method in model.config.ACCEPT_LOGLIKE:
        accept = ll > model.results.best_loglikelihood
    else:
        accept = error < model.results.best_valid_error

    if accept:
        max_patience = max(model.patience, i * model.patience_increase)
        model.patience = int(min(max_patience, model.max_iterations))

        now = perf_counter()
        accept_time = round(now - model.start_time, 3)
        model.results.accept_time = time_format(accept_time)

    return accept


def verbose_logging(model, gnorm, epoch, i, ll, error, lr=None):
    info_print = False
    acceptance_method = model.config.acceptance_method
    b_ll = model.results.best_loglikelihood

    if acceptance_method in model.config.ACCEPT_LOGLIKE:
        if ll > (b_ll / model.config.likelihood_threshold):
            info_print = True
    else:
        if error < (model.results.best_valid_error - 0.001):
            info_print = True

    if epoch == (model.max_epochs // 10):
        info_print = True
    if gnorm < (model.results.gnorm_min / 5.0):
        info_print = True
        model.results.gnorm_min = gnorm

    time_passed = perf_counter() - model.start_time
    if (time_passed % 300) < model.config.TIME_COUNTER:
        info_print = True
    model.config.TIME_COUNTER = time_passed % 300

    status = f"Train {hf(i)}/{hf(model.patience)} (epoch={hf(epoch)}, LL={ll:.2f}, error={error*100:.2f}%, gnorm={gnorm:.3e})"

    if lr is not None:
        status += f", lr={lr:.2e}"

    if info_print:
        info(status)
        info_print = False


def early_stopping(model, gnorm, i):
    threshold = model.convergence_threshold
    if gnorm < threshold:
        return [True, "gnorm < threshold"]
    if i > model.patience:
        return [True, "iteration > patience"]
    return [False, "0"]
