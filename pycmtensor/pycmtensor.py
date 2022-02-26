# pymctensor.py

import aesara
import aesara.tensor as aet
import biogeme.database as biodb
import dill as pickle
import numpy as np

from pycmtensor.expressions import Beta, Weights
from pycmtensor.functions import errors, full_loglikelihood
from pycmtensor.utils import learn_rate_tempering, tqdm_nb_check

floatX = aesara.config.floatX


class PyCMTensorModel:
    def __init__(self, db):
        self.name = "PyCMTensorModel"
        self.params = []  # keep track of params
        self.beta_params = []
        self.inputs = db.tensors()

    def append_to_params(self, params):
        assert isinstance(params, (dict, list)), f"params must be of type dict or list"
        if isinstance(params, list):
            d = {}
            for param in params:
                d[str(param)] = param
            params = d

        for _, param in params.items():
            if isinstance(param, (Beta, Weights)):
                # update model params into list
                self.params.append(param)
                if isinstance(param, (Beta)):
                    self.beta_params.append(param)

    def get_weights(self):
        return [p for p in self.params if ((p().ndim > 1) and (p.status != 1))]

    def get_weight_values(self):
        return [p() for p in self.params if (p().ndim > 1)]

    def get_weight_size(self):
        if len(self.get_weights()) > 0:
            fn = aesara.function(
                [], sum([aet.prod(w.shape) for w in self.get_weights()])
            )
            return fn()
        else:
            return 0

    def get_betas(self):
        return [p for p in self.beta_params if (p.status != 1)]

    def get_beta_values(self):
        return [p() for p in self.beta_params]

    def __repr__(self):
        return f"{self.name}"


class Database(biodb.Database):
    def __init__(self, name, pandasDatabase, choiceVar="CHOICE"):
        super().__init__(name, pandasDatabase)
        self.choiceVar = choiceVar
        for name, variable in self.variables.items():
            if name in self.data.columns:
                if name == choiceVar:
                    variable.y = aet.ivector(name)
                else:
                    variable.x = aet.vector(name)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getattr__(self, name):
        if name in self.variables:
            return self.variables[name]

    def __getitem__(self, item):
        """Returns the aesara.tensor.var.TensorVariable object.
        Use Database["columnName"] to reference the TensorVariable
        """
        if hasattr(self.variables[item], "x"):
            return self.variables[item].x
        elif hasattr(self.variables[item], "y"):
            return self.variables[item].y
        else:
            raise NotImplementedError(f"Variable {item} not found")

    def compile_data(self):
        self.sharedData = {}
        for name, variable in self.variables.items():
            if name in self.data.columns:
                shared_data = aesara.shared(
                    np.asarray(self.data[name], dtype=floatX), borrow=True
                )
                if name == self.choiceVar:
                    self.sharedData[name] = aet.cast(shared_data, "int32")
                else:
                    self.sharedData[name] = shared_data

    def get_x_tensors(self):
        x_tensors = []
        for _, variable in self.variables.items():
            if hasattr(variable, "x"):
                x_tensors.append(variable.x)
        return x_tensors

    def get_x_data(self, index=None, batch_size=None, shift=None):
        x_data = []
        x_tensors = self.get_x_tensors()
        for x_tensor in x_tensors:
            if index == None:
                x_data.append(self.data[x_tensor.name])
            else:
                start = index * batch_size + shift
                end = (index + 1) * batch_size + shift
                x_data.append(self.data[x_tensor.name][start:end])
        return x_data

    def get_y_tensors(self):
        y_tensors = []
        for _, variable in self.variables.items():
            if hasattr(variable, "y"):
                y_tensors.append(variable.y)
        return y_tensors

    def get_y_data(self, index=None, batch_size=None, shift=None):
        y_data = []
        y_tensors = self.get_y_tensors()
        for y_tensor in y_tensors:
            if index == None:
                y_data.append(self.data[y_tensor.name])
            else:
                start = index * batch_size + shift
                end = (index + 1) * batch_size + shift
                y_data.append(self.data[y_tensor.name][start:end])
        return y_data

    def tensors(self):
        return self.get_x_tensors() + self.get_y_tensors()

    def input_data(self, index=None, batch_size=None, shift=None):
        """Outputs a list of pandas table data corresponding to the
            Symbolic variables

        Args:
            index (int, optional): Starting index slice.
            batch_size (int, optional): Size of slice.
            shift (int, optional): Add a random shift of the index.

        Returns:
            list: list of database (input) data
        """
        return self.get_x_data(index, batch_size, shift) + self.get_y_data(
            index, batch_size, shift
        )

    def input_shared_data(self):
        """Outputs a list of sharedVariable data corresponding to the symbolic
        variables

        Returns:
            list: a list  of sharedVariable database (input) data
        """
        shared_data = []
        tensors = self.get_x_tensors() + self.get_y_tensors()
        for tensor in tensors:
            shared_data.append(self.sharedData[tensor.name])
        return shared_data

    def autoscale(self, variables=None, verbose=False, default=None):
        for d in self.data:
            max_val = np.max(self.data[d])
            min_val = np.min(self.data[d])
            scale = 1.0
            if variables is None:
                varlist = self.get_x_tensors()
            else:
                varlist = variables
            if d in varlist:
                if min_val >= 0.0:
                    if default is None:
                        while max_val > 10:
                            self.data[d] /= 10.0
                            scale *= 10.0
                            max_val = np.max(self.data[d])
                    else:
                        self.data[d] /= default
                        scale = default
                if verbose:
                    print("scaling {} by {}".format(d, scale))


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
    Model,
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
    assert issubclass(Model, PyCMTensorModel)
    db = database
    rng = np.random.default_rng(seed)
    model = build_functions(Model(db), db, optimizer)

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
                    model.best_ll_score = 1 - model.output_errors(*db.input_data())

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
