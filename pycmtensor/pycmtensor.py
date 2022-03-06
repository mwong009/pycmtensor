# pymctensor.py
""" Core functionality """

import timeit

import aesara
import aesara.tensor as aet
import dill as pickle
import numpy as np

from pycmtensor import logger as log
from pycmtensor.logger import PyCMTensorError

from .functions import errors, full_loglikelihood
from .models import PyCMTensorModel
from .trackers import IterationTracker
from .utils import tqdm_nb_check


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
    lr = aet.scalar("learning_rate")
    index = aet.lscalar("index")
    batch_size = aet.lscalar("batch_size")
    shift = aet.lscalar("shift")
    if optimizer is not None:
        db.compile_data()
        opt = optimizer(model.params)
        updates = opt.update(model.cost, model.params, lr)

        model.loglikelihood_estimation = aesara.function(
            inputs=[index, batch_size, shift, lr],
            outputs=model.cost,
            updates=updates,
            on_unused_input="ignore",
            givens=[
                (t, data[index * batch_size + shift : (index + 1) * batch_size + shift])
                for t, data in zip(model.inputs, db.input_shared_data())
            ],
        )

    model.loglikelihood = aesara.function(
        inputs=[],
        outputs=full_loglikelihood(model.p_y_given_x, model.y),
        on_unused_input="ignore",
        givens={t: data for t, data in zip(model.inputs, db.input_shared_data())},
        name="loglikelihood",
    )

    model.output_probabilities = aesara.function(
        inputs=[],
        outputs=model.p_y_given_x,
        on_unused_input="ignore",
        givens={t: data for t, data in zip(model.inputs, db.input_shared_data())},
        name="p_y_given_x",
    )

    model.output_predictions = aesara.function(
        inputs=[],
        outputs=model.pred,
        on_unused_input="ignore",
        givens={t: data for t, data in zip(model.inputs, db.input_shared_data())},
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
    end_time = timeit.default_timer()
    model.build_time = round(end_time - start_time, 3)
    return model


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


def train(
    model,
    database,
    optimizer,
    batch_size=256,
    max_epoch=2000,
    base_lr=0.01,
    seed=999,
    debug=False,
    notebook=False,
):
    """Default training algorithm. Returns the best model ``model`` object.

    Args:
        model (PyCMTensorModel): the ``model`` object to train.
        database (Database): the ``database`` object containing the data and tensor variables.
        optimizer (Optimizer): the type of optimizer to use to train the model.
        batch_size (`int`, optional): batch size per iteration. Defaults to 256.
        max_epoch (`int`, optional): maximum number of epochs to train. Defaults to 2000.
        base_lr (`float`, optional): the base learning rate to use. Defaults to 0.01.
        seed (`int`, optional): the random seed value. Defaults to 999.
        debug (`bool`, optional): outputs more verbosity if True. Defaults to False.
        notebook (`bool`, optional): set this flag to True if running on a `Jupyter Notebook <https://jupyter.org/>`_. Defaults to False.

    Returns:
        PyCMTensorModel: the output is a trained ``model`` object. Call :class:`~pycmtensor.results.Results` to generate model results.

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
    inspect_model(model)
    model = build_functions(model, database, optimizer)

    # model config
    model.config["seed"] = seed
    model.config["max_epoch"] = max_epoch
    model.config["batch_size"] = batch_size
    model.config["base_lr"] = base_lr

    # training state
    epoch = 0
    tqdm = tqdm_nb_check(notebook)
    rng = np.random.default_rng(seed)
    tracker = IterationTracker()

    # training hyperparameters
    n_samples = database.get_rows()
    step_size = n_samples // batch_size
    max_iter = max_epoch * step_size
    patience = model.config["patience"]
    patience_increase = model.config["patience_increase"]
    validation_threshold = model.config["validation_threshold"]
    validation_frequency = min(step_size, patience / 2)

    # flags
    done_looping = False
    early_stopping = False

    start_time = timeit.default_timer()
    log.info(f"Training model...")
    print(
        f"dataset: {database.name} (n={n_samples})\n"
        + f"batch size: {batch_size}\n"
        + f"iterations per epoch: {step_size}"
    )

    model.null_ll = model.loglikelihood()
    model.best_ll_score = 1 - model.output_errors()
    model.best_ll = model.null_ll
    best_model = model

    if debug is False:
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

    while (epoch < max_epoch) and (not done_looping):
        epoch = epoch + 1

        for batch_index in range(step_size):
            iter = (epoch - 1) * step_size + batch_index + 1

            lr = base_lr
            i = rng.integers(0, step_size)
            shift = rng.integers(0, batch_size)
            # train model
            model.loglikelihood_estimation(i, batch_size, shift, lr)

            # validation step
            if iter % validation_frequency == 0:
                ll = model.loglikelihood()
                ll_score = 1 - model.output_errors()
                tracker.add(iter, "full_ll", ll)
                tracker.add(iter, "score", ll_score)
                if ll > model.best_ll:
                    if ll > (model.best_ll / validation_threshold):
                        patience = max(patience, iter * patience_increase)
                        patience = min(patience, max_iter)

                    model.best_epoch = epoch
                    model.best_ll = ll
                    model.best_ll_score = ll_score

                    best_model = model

                    if debug is False:
                        pbar0.postfix[0]["ll"] = model.best_ll
                        pbar0.postfix[1]["sc"] = model.best_ll_score
                        pbar0.update()

            if debug is False:
                pbar.set_description("Epoch {0:4d}/{1}".format(epoch, max_epoch))
                pbar.set_postfix({"Patience": f"{iter / patience * 100:.0f}%"})
                pbar.update()

            if patience <= iter:
                done_looping = True
                early_stopping = True
                break

    end_time = timeit.default_timer()
    model.train_time = end_time - start_time
    model.epochs_per_sec = round(epoch / model.train_time, 3)
    model.iterations = iter
    model.tracker = tracker

    with open(model.name + ".pkl", "wb") as f:
        pickle.dump(model, f)  # save model to pickle

    if early_stopping:
        log.warning("Maximum patience reached. Early stopping...")
    print(
        (
            "Optimization complete with accuracy of {0:6.3f}%."
            " Max loglikelihood reached @ epoch {1}.\n"
        ).format(model.best_ll_score * 100.0, model.best_epoch)
    )
    if debug is False:
        pbar0.close()
        pbar.close()

    return best_model
    # [train-end]
