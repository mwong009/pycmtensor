# pymctensor.py

import logging
import timeit

import aesara
import aesara.tensor as aet
import dill as pickle
import numpy as np

from pycmtensor import logger as log
from pycmtensor.logger import PyCMTensorError

from .functions import errors, full_loglikelihood
from .models import PyCMTensorModel
from .utils import learn_rate_tempering, tqdm_nb_check


def build_functions(model, db, optimizer=None):
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

    model.output_choices = aesara.function(
        inputs=[],
        outputs=model.pred,
        on_unused_input="ignore",
        givens={t: data for t, data in zip(model.inputs, db.input_shared_data())},
        name="predict",
    )

    model.output_estimated_betas = aesara.function(
        inputs=[],
        outputs=model.get_beta_values(),
        on_unused_input="ignore",
    )

    model.output_estimated_weights = aesara.function(
        inputs=[],
        outputs=model.get_weight_values(),
        on_unused_input="ignore",
    )

    model.output_errors = aesara.function(
        inputs=[],
        outputs=errors(model.p_y_given_x, model.y),
        on_unused_input="ignore",
        givens={t: data for t, data in zip(model.inputs, db.input_shared_data())},
        name="errors",
    )
    end_time = timeit.default_timer()
    model.build_time = round(end_time - start_time, 3)
    return model


def inspect_model(model):
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

    # training hyperparameters
    n_samples = database.get_rows()
    step_size = n_samples // batch_size
    max_iter = max_epoch * step_size
    patience = max_iter // 2
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
            iter = (epoch - 1) * step_size + batch_index

            lr = base_lr
            i = rng.integers(0, step_size)
            shift = rng.integers(0, batch_size)
            # train model
            model.loglikelihood_estimation(i, batch_size, shift, lr)

            # validation step
            if (iter + 1) % validation_frequency == 0:
                ll = model.loglikelihood()
                if ll > model.best_ll:
                    if ll > (model.best_ll * validation_threshold):
                        patience = max(patience, iter * patience_increase)
                        patience = min(patience, max_iter)

                    model.best_epoch = epoch
                    model.best_ll_score = 1 - model.output_errors()
                    model.best_ll = ll

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
