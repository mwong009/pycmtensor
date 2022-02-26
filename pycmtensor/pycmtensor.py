# pymctensor.py

import aesara
import aesara.tensor as aet
import dill as pickle
import numpy as np

from pycmtensor.functions import errors, full_loglikelihood
from pycmtensor.models import PyCMTensorModel
from pycmtensor.utils import learn_rate_tempering, tqdm_nb_check


def build_functions(model, db, optimizer=None):
    print("Building model...")
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
    )

    model.output_probabilities = aesara.function(
        inputs=[],
        outputs=model.p_y_given_x,
        on_unused_input="ignore",
        givens={t: data for t, data in zip(model.inputs, db.input_shared_data())},
    )

    model.output_choices = aesara.function(
        inputs=[],
        outputs=model.pred,
        on_unused_input="ignore",
        givens={t: data for t, data in zip(model.inputs, db.input_shared_data())},
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
    )

    return model


def train(
    model,
    database,
    optimizer,
    batch_size=256,
    max_epoch=2000,
    lr_init=0.01,
    seed=999,
    debug=False,
    notebook=False,
):
    tqdm = tqdm_nb_check(notebook)
    assert isinstance(model, PyCMTensorModel), f"{model} is an invalid model."
    db = database
    rng = np.random.default_rng(seed)
    model = build_functions(model, db, optimizer)

    n_samples = len(db.data)
    n_batches = n_samples // batch_size
    patience = 20000
    patience_increase = 2
    validation_threshold = 1.003
    validation_frequency = min(n_batches, patience / 2)
    done_looping = False
    early_stopping = False
    total_iter = max_epoch * n_batches
    epoch = 0

    print("dataset: {} ({})".format(db.name, n_samples))
    print("batch size: {}".format(batch_size))
    print("batches per epoch: {}".format(n_batches))
    print("validation frequency: {}\n".format(validation_frequency))
    print("Training model...")

    model.null_ll = model.loglikelihood()
    model.best_ll_score = 1 - model.output_errors()
    model.best_ll = model.null_ll

    if debug is False:
        pbar0 = tqdm(
            bar_format=(
                "Loglikelihood:  {postfix[0][ll]:.6f}  Score: {postfix[1][sc]:.3f}"
            ),
            postfix=[{"ll": model.null_ll}, {"sc": model.best_ll_score}],
            position=0,
            leave=True,
        )
        pbar = tqdm(
            total=total_iter,
            desc="Epoch {0:4d}/{1}".format(0, total_iter),
            unit_scale=True,
            position=1,
            leave=True,
        )

    while (epoch < max_epoch) and (not done_looping):
        epoch = epoch + 1

        for batch_index in range(n_batches):
            iter = (epoch - 1) * n_batches + batch_index

            lr = learn_rate_tempering(iter, patience, lr_init)
            i = rng.integers(0, n_batches)
            shift = rng.integers(0, batch_size)
            # train model
            model.loglikelihood_estimation(i, batch_size, shift, lr)

            # validation step
            if (iter + 1) % validation_frequency == 0:
                ll = model.loglikelihood()
                if ll > model.best_ll:
                    model.best_ll = ll
                    model.best_epoch = epoch
                    model.best_ll_score = 1 - model.output_errors()

                    if debug is False:
                        pbar0.postfix[0]["ll"] = model.best_ll
                        pbar0.postfix[1]["sc"] = model.best_ll_score
                        pbar0.update()

                    if ll > (model.best_ll * validation_threshold):
                        patience = max(patience, iter * patience_increase)
                        best_model = model

            if debug is False:
                pbar.set_description("Epoch {0:4d}/{1}".format(epoch, max_epoch))
                pbar.set_postfix({"Patience": f"{iter / patience * 100:.0f}%"})
                pbar.update()

            if patience <= iter:
                done_looping = True
                early_stopping = True
                break

    with open(model.name + ".pkl", "wb") as f:
        pickle.dump(model, f)  # save model to pickle

    if early_stopping:
        print("Maximum patience reached. Early stopping...")
    print(
        (
            "Optimization complete with accuracy of {0:6.3f}%\n"
            " with maximum loglikelihood reached @ epoch {1}."
        ).format(model.best_ll_score * 100.0, model.best_epoch)
    )
    if debug is False:
        pbar0.close()
        pbar.close()

    return best_model
