from collections import OrderedDict
from time import perf_counter

import numpy as np

from pycmtensor.expressions import Beta
from pycmtensor.logger import debug, info, warning
from pycmtensor.utils import human_format as hf
from pycmtensor.utils import time_format

IS_TRAINING = 1


def compute(model, ds, update=False, **params):
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
        model.lr_scheduler = model.config.lr_scheduler(
            lr=model.config.base_learning_rate,
            max_lr=model.config.max_learning_rate,
            max_epochs=model.config.max_epochs,
            factor=model.config.lr_stepLR_factor,
            drop_every=model.config.lr_stepLR_drop_every,
            power=model.config.lr_PolynomialLR_power,
            cycle_steps=model.config.lr_CLR_cycle_steps,
            gamma=model.config.lr_ExpRangeCLR_gamma,
        )
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


def train(model, ds, **kwargs):
    """Main training loop

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

    model.patience = max(model.config.patience, n_train_batches)
    validation_freq = n_train_batches
    max_epochs = model.config.max_epochs
    model.config.max_iterations = max_epochs * n_train_batches

    done = False
    epoch = 0
    i = 0
    shift = 0

    optimizer = model.config.optimizer(model.params, config=model.config)
    updates = optimizer.update(model.cost, model.params, model.learning_rate)
    model.build_cost_updates_fn(updates)

    model.lr_scheduler = model.config.lr_scheduler(
        lr=model.config.base_learning_rate,
        max_lr=model.config.max_learning_rate,
        max_epochs=model.config.max_epochs,
        factor=model.config.lr_stepLR_factor,
        drop_every=model.config.lr_stepLR_drop_every,
        power=model.config.lr_PolynomialLR_power,
        cycle_steps=model.config.lr_CLR_cycle_steps,
        gamma=model.config.lr_ExpRangeCLR_gamma,
    )

    model.results.statistics_graph = {
        "train_ll": [],
        "train_error": [],
        "valid_error": [],
    }

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
    model.results.n_train = n_train
    model.results.n_valid = n_valid
    model.results.n_params = model.n_params
    model.results.config = model.config
    model.results.converged = [False, "0"]

    params_prev = initialize_results(model, train_data, t_index)
    batch_data = []
    for batch in range(n_train_batches):
        batch_data.append(ds.train_dataset(x_y, batch, batch_size, shift))

    info(
        f"Start (n={n_train}, epoch={epoch}, NLL={model.results.null_loglikelihood:.2f}, error={model.results.best_valid_error*100:.2f}%)"
    )

    model.results.start_time = perf_counter()
    while (epoch < model.config.max_epochs) and (not done):
        learning_rate = model.lr_scheduler(epoch)  # get learning rate

        for batch in range(n_train_batches):
            index = np.arange(len(batch_data[batch][-1]))
            model.cost_updates_fn(*batch_data[batch], learning_rate, index, IS_TRAINING)

            if i % validation_freq == 0:
                # train step
                log_like, train_error, valid_error = train_step(
                    model, ds, train_data, t_index, valid_data
                )
                append_statistics(model, log_like, train_error, valid_error)

                # convergence
                gnorm, params_prev = gnorm_func(model, train_data, t_index, params_prev)

                # verbose logging
                verbose_logging(model, gnorm, epoch, i, log_like, valid_error)

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
            if done_looping(model, epoch, i, log_like, valid_error, gnorm):
                done = True
                break

        epoch += 1  # increment epoch

    # post training process
    model = post_training(model, epoch, i)

    # Save hessian data
    model.save_hessian_data(n_train, train_data)


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


def append_statistics(model, log_like, train_error, valid_error):
    model.results.statistics_graph["train_ll"].append(log_like)
    model.results.statistics_graph["valid_error"].append(valid_error)
    model.results.statistics_graph["train_error"].append(train_error)


def done_looping(model, epoch, i, log_like, error, gnorm):
    if (i > model.patience) or (model.results.converged[0] != False):
        info(
            f"Train {hf(i-1)}/{hf(model.patience)} (epoch={hf(epoch)}, LL={log_like:.2f}, error={error*100:.2f}%, gnorm={gnorm:.3e})"
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
    accept = False
    acceptance_method = model.config.acceptance_method

    if acceptance_method in model.config.ACCEPT_LOGLIKE:
        accept = ll > model.results.best_loglikelihood
    else:
        accept = error < model.results.best_valid_error

    if accept:
        max_patience = max(model.patience, i * model.config.patience_increase)
        model.patience = int(min(max_patience, model.config.max_iterations))

    return accept


def verbose_logging(model, gnorm, epoch, i, ll, error):
    info_print = False
    acceptance_method = model.config.acceptance_method
    b_ll = model.results.best_loglikelihood

    if acceptance_method in model.config.ACCEPT_LOGLIKE:
        if ll > (b_ll / model.config.likelihood_threshold):
            info_print = True
    else:
        if error < (model.results.best_valid_error - 0.001):
            info_print = True

    if epoch == (model.config.max_epochs // 10):
        info_print = True
    if gnorm < (model.results.gnorm_min / 5.0):
        info_print = True
        model.results.gnorm_min = gnorm

    time_passed = perf_counter() - model.results.start_time
    if (time_passed % 300) < model.config.TIME_COUNTER:
        info_print = True
    model.config.TIME_COUNTER = time_passed % 300

    if info_print:
        info(
            f"Train {hf(i)}/{hf(model.patience)} (epoch={hf(epoch)}, LL={ll:.2f}, error={error*100:.2f}%, gnorm={gnorm:.3e})"
        )
        info_print = False


def early_stopping(model, gnorm, i):
    threshold = model.config.convergence_threshold
    if gnorm < threshold:
        return [True, "gnorm < threshold"]
    if i > model.patience:
        return [True, "iteration > patience"]
    return [False, "0"]


def post_training(model, epoch, i):
    now = perf_counter()
    train_time = round(now - model.results.start_time, 3)
    model.results.train_time = time_format(train_time)
    model.results.epochs_per_sec = round(epoch / train_time, 2)

    threshold = model.config.convergence_threshold
    if model.results.converged[1] == "gnorm < threshold":
        info(f"Model converged (t={train_time}), gnorm < {threshold}")
    else:
        info(f"Max iters reached: {hf(i-1)}/{hf(model.patience)} (t={train_time})")

    for key, value in model.results.statistics_graph.items():
        model.results.statistics_graph[key] = np.array(value).tolist()
    model.results.statistics_graph["learning_rate"] = model.lr_scheduler.history

    best_epoch = model.results.best_epoch
    best_ll = model.results.best_loglikelihood
    best_error = model.results.best_valid_error
    gnorm = model.results.gnorm

    info(
        f"Best results obtained at epoch {best_epoch}: LL={best_ll:.2f}, error={best_error*100:.2f}%, gnorm={gnorm:.5e}"
    )

    return model
