# pymctensor.py
""" Core functionality """

import timeit

import aesara
import aesara.tensor as aet
import numpy as np

from pycmtensor import logger as log
from pycmtensor import scheduler as schlr

from .functions import bhhh, errors, full_loglikelihood, gradient_norm, hessians
from .logger import PyCMTensorError
from .models import PyCMTensorModel
from .scheduler import CyclicLR
from .trackers import IterationTracker
from .utils import inspect_model, save_to_pickle


def train(model, database, optimizer, save_model=False, **kwargs):
    """Default training algorithm. Returns the best model ``model`` object.

    Args:
        model (PyCMTensorModel): the ``model`` object to train.
        database (Database): the ``database`` object containing the data and tensor
        variables.
        optimizer (Optimizer): the type of optimizer to use to train the model.
        save_model (bool): flag for saving model to a pickle file (disabled currently
        because buggy)

    Returns:
        PyCMTensorModel: the output is a trained ``model`` object. Call :class:`~pycmtensor.results.Results` to generate model results.

    Note:
        ``**kwargs`` can be any of the following: 'patience', 'patience_increase',
        'validation_threshold', 'seed', 'base_lr', 'max_lr', 'batch_size',
        'max_epoch', 'debug', 'notebook', 'learning_scheduler', 'cyclic_lr_mode',
        'cyclic_lr_step_size'. See config.py for more.

    Example:
        .. code-block :: python

            import pycmtensor as cmt
            from pycmtensor.models import MNLModel
            from pycmtensor.optimizers import Adam
            db = cmt.Database(pandasDatabase=some_pandas_data)
            ...
            model = MNLogit(u=U, av=AV, database=db, name="mymodel")
            model = cmt.train(model, database=db, optimizer=Adam)
            ...
    """

    # [train-start]
    print("Python", model.config["python_version"])

    # pre-run routine #
    inspect_model(model)
    model.build_functions(database, optimizer)

    # load kwargs into model.config() #
    for key, val in kwargs.items():
        if key in model.config():
            if type(val) != type(model.config[key]):
                raise TypeError(
                    f"{key}={val} must be of type {type(model.config[key])}"
                )
            model.config[key] = val
        else:
            raise NotImplementedError(
                f"Invalid option in kwargs {key}={val}\n"
                + "Valid options are: {model.config}"
            )

    # load learning rate scheduler #
    if model.config["learning_scheduler"] in schlr.__dict__:
        Scheduler = getattr(schlr, model.config["learning_scheduler"])
    else:
        raise NotImplementedError(
            f"Invalid option for learning_scheduler: {model.config['learning_scheduler']}"
        )

    # create learning rate scheduler #
    scheduler_kwargs = {"base_lr": model.config["base_lr"]}
    if Scheduler == CyclicLR:
        scheduler_kwargs.update(
            {
                "max_lr": np.maximum(model.config["base_lr"], model.config["max_lr"]),
                "step_size": model.config["cyclic_lr_step_size"],
                "mode": model.config["cyclic_lr_mode"],
            }
        )
    lr_scheduler = Scheduler(**scheduler_kwargs)

    # model training hyperparameters #
    batch_size = model.config["batch_size"]
    patience = model.config["patience"]
    patience_increase = model.config["patience_increase"]
    validation_threshold = model.config["validation_threshold"]
    n_samples = database.get_rows()
    n_batches = n_samples // batch_size
    max_epoch = model.config["max_epoch"]
    max_iter = max_epoch * n_batches
    if max_epoch < int(patience / n_batches):
        patience = max_iter  # clamp patience to maximum iterations
    validation_frequency = min(n_batches, patience / 2)

    # set inital model solutions #
    model.null_ll = model.loglikelihood()
    model.best_ll_score = 1 - model.output_errors()
    model.best_ll = model.null_ll
    best_model = model

    # verbosity #
    vb = model.config["verbosity"]
    debug = model.config["debug"]

    log.info(f"Training model...")
    # print(
    #     f"\n"
    #     + f"dataset: {database.name} (n={n_samples})\n"
    #     + f"batch size: {batch_size}\n"
    #     + f"iterations per epoch: {n_batches}\n"
    # )

    # training states and trackers #
    done_looping = False
    early_stopping = False
    epoch = 0
    iter = 0
    last_logged = 0
    track_index = 0
    tracker = IterationTracker(iterations=max_iter)
    rng = np.random.default_rng(model.config["seed"])
    start_time = timeit.default_timer()

    # training run loop #
    while (epoch < max_epoch) and (not done_looping):

        epoch = epoch + 1  # increment epoch
        if epoch < max_epoch // 2:  # set the learning rate for this epoch
            epoch_lr = lr_scheduler.get_lr(epoch)

        for _ in range(n_batches):  # loop over n_batches
            i = rng.integers(0, n_batches)  # select random index and shift slices
            shift = rng.integers(0, batch_size)

            # train model step #
            model.loglikelihood_estimation(i, batch_size, shift, epoch_lr)

            # validation step, validate every `validation_frequency` #
            if iter % validation_frequency == 0:
                ll = model.loglikelihood()  # record the loglikelihood
                ll_score = 1 - model.output_errors()  # record the score

                # track the progress of the training into the tracker log
                tracker.add(track_index, "full_ll", ll)
                tracker.add(track_index, "score", ll_score)
                tracker.add(track_index, "lr", epoch_lr)
                track_index += 1

                # update the best model #
                if ll > model.best_ll:
                    if debug and (epoch > (last_logged + 10)):
                        log.info(
                            f"{epoch:4} log likelihood {ll:.2f} | score {ll_score:.2f} | learning rate {epoch_lr:.2e}"
                        )
                        last_logged = epoch
                    if ll > (model.best_ll / validation_threshold):
                        patience = min(
                            max(patience, iter * patience_increase), max_iter
                        )

                    # record training statistics #
                    model.best_epoch = epoch
                    model.best_ll = ll
                    model.best_ll_score = ll_score

                    best_model = model

            if patience <= iter:
                done_looping = True
                early_stopping = True
                break

            iter += 1  # increment iteration

    # end of training sequence #
    end_time = timeit.default_timer()
    best_model.train_time = end_time - start_time
    best_model.epochs_per_sec = round(epoch / model.train_time, 3)
    best_model.iter_per_sec = round(iter / model.train_time, 3)
    best_model.iterations = iter
    best_model.tracker = tracker

    if save_model:
        save_to_pickle(best_model)

    if early_stopping:
        log.info("Maximum iterations reached. Terminating...")

    score = best_model.best_ll_score * 100.0
    log.info(f"Optimization complete with accuracy of {score:.3f}%.")
    log.info(f"Max log likelihood reached @ epoch {best_model.best_epoch}.")

    return best_model
    # [train-end]
