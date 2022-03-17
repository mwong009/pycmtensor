# pymctensor.py
""" Core functionality """

import timeit

import aesara
import aesara.tensor as aet
import dill as pickle
import numpy as np

from pycmtensor import logger as log
from pycmtensor import scheduler as schlr

from .functions import bhhh, errors, full_loglikelihood, gradient_norm, hessians
from .logger import PyCMTensorError
from .models import PyCMTensorModel
from .scheduler import CyclicLR
from .trackers import IterationTracker
from .utils import save_to_pickle, tqdm_nb_check


def build_functions(model, db, optimizer=None):
    """Build callable objects that will calculate ``outputs`` from ``inputs``.

    Args:
        model (PyCMTensorModel): must be of a :class:`PyCMTensorModel` model class object.
        db (Database): the :class:`Database` object.
        optimizer (Optimizer, optional): optimizer object class to use. If ``None`` is given, skips the build of loglikelihood_estimation function.

    Returns:
        PyCMTensorModel: the updated ``model`` instance.

    Note:
        :func:`build_functions` is called internally from :func:`train`. Generally you do not need to call this in your program.
    """
    log.info("Building model...")
    start_time = timeit.default_timer()
    if optimizer is not None:
        lr = aet.scalar("learning_rate")
        index = aet.lscalar("index")
        batch_size = aet.lscalar("batch_size")
        shift = aet.lscalar("shift")
        db.compile_data()
        opt = optimizer(model.params)
        updates = opt.update(model.cost, model.params, lr)

        model.loglikelihood_estimation = aesara.function(
            inputs=[index, batch_size, shift, lr],
            outputs=model.cost,
            updates=updates,
            on_unused_input="ignore",
            givens={
                t: data[index * batch_size + shift : (index + 1) * batch_size + shift]
                for t, data in zip(model.inputs, db.input_shared_data())
            },
        )

    model.loglikelihood = aesara.function(
        inputs=[],
        outputs=full_loglikelihood(model.p_y_given_x, model.y),
        on_unused_input="ignore",
        givens={t: data for t, data in zip(model.inputs, db.input_shared_data())},
        name="loglikelihood",
    )

    model.output_probabilities = aesara.function(
        inputs=model.inputs,
        outputs=model.prob().T,  # returns (N x J) matrix
        on_unused_input="ignore",
        name="p_y_given_x",
    )

    model.output_predictions = aesara.function(
        inputs=model.inputs,
        outputs=model.pred.T,  # returns (N x 1) matrix
        on_unused_input="ignore",
        name="output_predictions",
    )

    model.output_estimated_betas = aesara.function(
        inputs=[],
        outputs=model.get_beta_values(),
        on_unused_input="ignore",
        name="output_betas",
    )

    model.output_estimated_weights = aesara.function(
        inputs=[],
        outputs=model.get_weight_values(),
        on_unused_input="ignore",
        name="output_weights",
    )

    model.output_errors = aesara.function(
        inputs=[],
        outputs=errors(model.p_y_given_x, model.y),
        on_unused_input="ignore",
        givens={t: data for t, data in zip(model.inputs, db.input_shared_data())},
        name="output_errors",
    )
    model.H = aesara.function(
        inputs=[],
        outputs=hessians(model.p_y_given_x, model.y, model.beta_params),
        on_unused_input="ignore",
        givens={t: data for t, data in zip(model.inputs, db.input_shared_data())},
    )

    model.BHHH = aesara.function(
        inputs=[],
        outputs=bhhh(model.p_y_given_x, model.y, model.beta_params),
        on_unused_input="ignore",
        givens={t: data for t, data in zip(model.inputs, db.input_shared_data())},
    )

    model.gnorm = aesara.function(
        inputs=[],
        outputs=gradient_norm(model.p_y_given_x, model.y, model.beta_params),
        on_unused_input="ignore",
        givens={t: data for t, data in zip(model.inputs, db.input_shared_data())},
    )
    end_time = timeit.default_timer()
    model.build_time = round(end_time - start_time, 3)
    # return model


def inspect_model(model):
    """Raises and error if `model` is not a valid ``PyCMTensorModel`` class.

    Args:
        model (PyCMTensorModel): the constructed model class.

    Raises:
        PyCMTensorError: logs an error if the model class is an invalid class.

    Returns:
        PyCMTensorModel: Returns the ``model`` object.

    Example:
        .. code-block :: python

            import pycmtensor as cmt
            from pycmtensor.models import MNLModel
            db = cmt.Database(pandasDatabase=some_pandas_data)
            ...

            model = MNLogit(u=U, av=AV, database=db, name="mymodel")
            inpect_model(model)

    """
    if not isinstance(model, PyCMTensorModel):
        msg = f"{model} is not a valid PyCMTensorModel model."
        log.error(msg)
        raise PyCMTensorError(msg)
    return model


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
    # pre-run routine
    inspect_model(model)
    build_functions(model, database, optimizer)

    # load kwargs into model.config()
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

    # load learning rate scheduler
    if model.config["learning_scheduler"] in schlr.__dict__:
        Scheduler = getattr(schlr, model.config["learning_scheduler"])
    else:
        raise NotImplementedError(
            f"Invalid option for learning_scheduler: {model.config['learning_scheduler']}"
        )

    # create learning rate scheduler
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

    # training hyperparameters
    batch_size = model.config["batch_size"]
    patience = model.config["patience"]
    patience_increase = model.config["patience_increase"]
    validation_threshold = model.config["validation_threshold"]
    n_samples = database.get_rows()
    n_batches = n_samples // batch_size
    max_epoch = max(model.config["max_epoch"], int(patience / n_batches))
    if model.config["max_epoch"] < int(patience / n_batches):
        log.warning(
            f"max_epoch={model.config['max_epoch']} is smaller than expected value "
            f"={int(patience / n_batches)}, setting default max_epoch={max_epoch}."
        )
    max_iter = max_epoch * n_batches
    validation_frequency = min(n_batches, patience / 2)

    # flags
    done_looping = False
    early_stopping = False

    # disables verbose printout if debig is true
    if model.config["debug"] is False:
        log.info(f"Training model...")
        print(
            f"dataset: {database.name} (n={n_samples})\n"
            + f"batch size: {batch_size}\n"
            + f"iterations per epoch: {n_batches}"
        )

    # set inital model solutions
    model.null_ll = model.loglikelihood()
    model.best_ll_score = 1 - model.output_errors()
    model.best_ll = model.null_ll
    best_model = model

    # disables tqdm if debug is True
    if model.config["debug"] is False:
        tqdm = tqdm_nb_check(model.config["notebook"])
        pbar0 = tqdm(
            bar_format=(
                "Loglikelihood:  {postfix[0][ll]:.3f}  Score: {postfix[1][sc]:.3f}"
            ),
            postfix=[{"ll": model.null_ll}, {"sc": model.best_ll_score}],
            position=0,
            leave=True,
        )
        pbar = tqdm(
            total=max_iter,
            desc="Epoch {0:4d}/{1}".format(0, max_iter),
            unit_scale=True,
            position=1,
            leave=True,
        )

    # training state and tracker
    epoch = 0
    iter = 0
    track_index = 0
    tracker = IterationTracker(iterations=max_iter)
    rng = np.random.default_rng(model.config["seed"])
    start_time = timeit.default_timer()

    # training run step
    while (epoch < max_epoch) and (not done_looping):

        epoch = epoch + 1  # increment epoch
        if epoch < max_epoch // 2:  # set the learning rate for this epoch
            epoch_lr = lr_scheduler.get_lr(epoch)

        for _ in range(n_batches):  # loop over n_batches
            i = rng.integers(0, n_batches)  # select random index and shift slices
            shift = rng.integers(0, batch_size)

            # train model
            model.loglikelihood_estimation(i, batch_size, shift, epoch_lr)

            # validation step
            if iter % validation_frequency == 0:
                ll = model.loglikelihood()  # record the loglikelihood
                ll_score = 1 - model.output_errors()  # record the score

                # track the progress of the training into the tracker log
                tracker.add(track_index, "full_ll", ll)
                tracker.add(track_index, "score", ll_score)
                tracker.add(track_index, "lr", epoch_lr)
                track_index += 1

                # update the best model
                if ll > model.best_ll:
                    if ll > (model.best_ll / validation_threshold):
                        log.debug(
                            f"epoch {epoch} log likelihood {ll} score {ll_score} learning rate{epoch_lr}"
                        )
                        patience = max(patience, iter * patience_increase)
                        patience = min(patience, max_iter)

                    model.best_epoch = epoch
                    model.best_ll = ll
                    model.best_ll_score = ll_score

                    # update tqdm if debug is False
                    if model.config["debug"] is False:
                        pbar0.postfix[0]["ll"] = model.best_ll
                        pbar0.postfix[1]["sc"] = model.best_ll_score
                        pbar0.update()

                    best_model = model

            # update tqdm if debug is False
            if model.config["debug"] is False:
                pbar.set_description("Epoch {0:4d}/{1}".format(epoch, max_epoch))
                pbar.set_postfix({"Patience": f"{iter / patience * 100:.0f}%"})
                pbar.update()

            if patience <= iter:
                done_looping = True
                early_stopping = True
                break

            iter += 1  # increment iteration

    # end of training step
    end_time = timeit.default_timer()
    best_model.train_time = end_time - start_time
    best_model.epochs_per_sec = round(epoch / model.train_time, 3)
    best_model.iter_per_sec = round(iter / model.train_time, 3)
    best_model.iterations = iter
    best_model.tracker = tracker

    if save_model:
        save_to_pickle(best_model)

    if best_model.config["debug"] is False:
        if early_stopping:
            log.warning("Maximum patience reached. Early stopping...")
        print(
            (
                "Optimization complete with accuracy of {0:6.3f}%."
                " Max loglikelihood reached @ epoch {1}.\n"
            ).format(best_model.best_ll_score * 100.0, best_model.best_epoch)
        )

    # update tqdm if debug is False
    if best_model.config["debug"] is False:
        pbar0.close()
        pbar.close()

    return best_model
    # [train-end]
