# # pycmtensor.py
# """PyCMTensor main module"""
# from collections import OrderedDict
# from time import perf_counter
# from typing import Union

# import aesara.tensor as aet
# import numpy as np
# from aesara import function
# from aesara.tensor.var import TensorVariable

# from pycmtensor import config

# from .expressions import Beta, ExpressionParser, Param
# from .functions import errors, gradient_vector, hessians
# from .logger import debug, info, warning
# from .results import Results
# from .utils import time_format


# class PyCMTensorModel:
#     def __init__(self, db, **kwargs):
#         """Base model class object"""
#         self.name = "PyCMTensorModel"
#         self.config = config
#         self.rng = np.random.default_rng(config.seed)
#         self.params = []  # keep track of all the Params
#         self.betas = []  # keep track of the Betas
#         # self.weights = []  # keep track of the Weights
#         self.updates = []  # keep track of the updates
#         self.inputs = db.x
#         self.outputs = db.y
#         self.learning_rate = aet.scalar("learning_rate")
#         self.results = Results()

#         for key, value in kwargs.items():
#             self.config.add(key, value)

#         debug(f"Building model...")

#     # def add_params(self, params: Union[dict, list]):
#     #     """Method to load locally defined variables

#     #     Args:
#     #         params (Union[dict, list]): a dict or list of ``TensorSharedVariable``
#     #     """
#     #     if not isinstance(params, (dict, list)):
#     #         # raise error if params is not a dictionary or a list
#     #         raise TypeError(f"params must be of Type dict or list")

#     #     if isinstance(params, list):

#     #         for p in params:
#     #             if not isinstance(p, Param):
#     #                 # raise error if p in params is not a Param instance
#     #                 raise TypeError(f"{p} must of a Param instance. Returned type {type(p)}")

#     #         # create a dict of str(param): param, if params is given as a list
#     #         params = {str(p): p for p in params}
#     #         print(params)

#     #     if not hasattr(self, "cost"):
#     #         # raise error if not valid cost function defined
#     #         raise ValueError(f"No valid cost function defined.")

#     #     # get symbols from the cost function
#     #     symbols = ExpressionParser().parse(getattr(self, "cost"))
#     #     seen = set()

#     #     for _, p in params.items():

#     #         if (not isinstance(p, Param)) or (p.name in seen):
#     #             # skip if p is not a Param instance or p is already in self.params
#     #             continue

#     #         if p.name not in symbols:
#     #             # raise a warning if p is not in any utility function
#     #             warning(f"{p.name} not in any utility functions")
#     #             continue

#     #         # add p to self.params
#     #         self.params.append(p)
#     #         seen.add(p.name)

#     #         if isinstance(p, (Beta)):
#     #             # add pp to self.betas if p is a Beta instance
#     #             self.betas.append(p)
#     #             continue

#     def add_regularizers(self, l_reg: TensorVariable):
#         """Adds regularizer to cost function

#         Args:
#             l_reg (TensorVariable): symbolic variable defining the regularizer term
#         """
#         if not hasattr(self, "cost"):
#             # raise error if not valid cost function defined
#             raise ValueError("No valid cost function defined.")

#         self.cost += l_reg

#     @property
#     def n_params(self):
#         """Get the total number of parameters"""
#         return self.n_betas  # + sum(self.n_weights)

#     @property
#     def n_betas(self):
#         """Get the count of Beta parameters"""
#         return len(self.betas)

#     # @property
#     # def n_weights(self):
#     #     """Get the total number of elements of each weight matrix"""
#     #     return [w.size for w in self.get_weights()]

#     def get_betas(self) -> dict:
#         """Returns the Beta (key, value) pairs as a dict"""
#         return {beta.name: beta.get_value() for beta in self.betas}

#     # def get_weights(self) -> list[np.ndarray]:
#     #     """Returns the Weights as a list of matrices"""
#     #     return [w.get_value() for w in self.weights]

#     def reset_values(self):
#         """Resets Model parameters to their initial value"""
#         for p in self.params:
#             p.reset_value()

#     def model_update_wrt_cost(self):
#         """Loads the function to ``self.update_wrt_cost()`` to output the cost
#         and updates the model parameters given inputs and learning rate"""
#         self.update_wrt_cost = function(
#             name="update_wrt_cost",
#             inputs=self.inputs + [self.outputs] + [self.learning_rate],
#             outputs=self.cost,
#             updates=self.updates,
#         )

#     def model_loglikelihood(self):
#         """Loads the function to ``self.loglikelihood()`` to output the loglikelihood
#         value of the model given inputs"""
#         self.loglikelihood = function(
#             name="loglikelihood",
#             inputs=self.inputs + [self.outputs],
#             outputs=self.ll,
#         )

#     def model_choice_probabilities(self):
#         """Loads the function to ``self.choice_probabilities()`` to output discrete
#         choice probabilities. Axes of outputs are swapped"""
#         self.choice_probabilities = function(
#             name="choice probabilities",
#             inputs=self.inputs,
#             outputs=self.p_y_given_x.swapaxes(0, 1),
#         )

#     def model_choice_predictions(self):
#         """Loads the function to ``self.choice_predictions()`` to output discrete
#         choice predictions"""
#         self.choice_predictions = function(
#             name="choice predictions",
#             inputs=self.inputs,
#             outputs=self.pred,
#         )

#     def model_prediction_error(self):
#         """Loads the function to ``self.prediction_error()`` to output the model error
#         wrt inputs"""
#         self.prediction_error = function(
#             name="prediction error",
#             inputs=self.inputs + [self.outputs],
#             outputs=errors(self.p_y_given_x, self.outputs),
#         )

#     def model_H(self):
#         """Loads the function to ``self.H()`` to calculate the Hessian matrix or the
#         2nd-order partial derivatives of the model.
#         """
#         self.H = function(
#             name="Hessian matrix",
#             inputs=self.inputs + [self.outputs],
#             outputs=hessians(self.cost, self.betas),
#             allow_input_downcast=True,
#         )

#     def model_G(self):
#         """Loads the function to ``self.G()`` to calculate the first order gradients
#         of the model.
#         """
#         self.G = function(
#             name="BHHH matrix",
#             inputs=self.inputs + [self.outputs],
#             outputs=gradient_vector(self.cost, self.betas),
#             allow_input_downcast=True,
#         )

#     def predict(self, db, return_choices: bool = True):
#         """Returns the predicted choice or choice probabilites

#         Args:
#             db (pycmtensor.Data): database for prediction
#             return_choices (bool): if `True` then returns discrete choices instead of
#                 probabilities

#         Returns:
#             numpy.ndarray: the output vector
#         """
#         if not return_choices:
#             return self.choice_probabilities(*db.valid(self.inputs))
#         else:
#             return self.choice_predictions(*db.valid(self.inputs))

#     def train(self, db, **kwargs):
#         """Function to train the model

#         Args:
#             db (pycmtensor.Data): database used to train the model
#             **kwargs: keyword arguments for adjusting training configuration.
#                 Possible values are `max_steps:int`, `patience:int`,
#                 `lr_scheduler:scheduler.Scheduler`, `batch_size:int`. For more
#                 information and other possible options, see
#                 :py:data:`hyperparameters <pycmtensor.config.Config>`
#         """
#         for key, value in kwargs.items():
#             self.config.add(key, value)

#         # [train-start]
#         batch_size = self.config.batch_size
#         if (batch_size == 0) or (batch_size == None):
#             batch_size = db.n_train

#         db.n_train_batches = db.n_train // batch_size

#         patience = max(self.config.patience, db.n_train_batches)
#         patience_inc = self.config.patience_increase
#         max_steps = self.config.max_steps
#         validation_threshold = self.config.validation_threshold
#         convergence_threshold = self.config.convergence_threshold

#         validation_freq = db.n_train_batches
#         max_steps = max_steps * db.n_train_batches

#         lr_scheduler = self.config.lr_scheduler(
#             lr=self.config.base_learning_rate,
#             factor=self.config.lr_stepLR_factor,
#             drop_every=self.config.lr_stepLR_drop_every,
#             power=self.config.lr_PolynomialLR_power,
#             cycle_steps=self.config.lr_CLR_cycle_steps,
#             gamma=self.config.lr_ExpRangeCLR_gamma,
#         )

#         # initalize results array
#         self.results.performance_graph = OrderedDict()

#         # compute the inital results of the model
#         x_and_y = self.inputs + [self.outputs]
#         self.results.init_loglikelihood = self.loglikelihood(*db.train(x_and_y))
#         self.results.best_valid_error = self.prediction_error(*db.valid(x_and_y))
#         self.results.best_loglikelihood = self.results.init_loglikelihood
#         self.results.betas = self.betas

#         # loop parameters
#         done_looping = False
#         converged = False
#         step = 0
#         iteration = 0
#         shift = 0
#         fg_norm_tol = np.inf

#         betas_prev = [b.get_value() for b in self.results.betas]

#         # set learning rate
#         learning_rate = lr_scheduler(step)

#         # main loop
#         start_time = perf_counter()
#         info(
#             f"Start (n={db.n_train}, Step={step}, LL={self.results.null_loglikelihood:.2f}, Error={self.results.best_valid_error*100:.2f}%)"
#         )

#         while (step < max_steps) and (not done_looping):
#             # increment step
#             step += 1

#             # loop over batch
#             learning_rate = lr_scheduler(step)
#             for index in range(db.n_train_batches):
#                 if (iteration > patience) or converged:
#                     step -= 1
#                     info(
#                         f"Train (Step={step}, LL={train_loglike:.2f}, Error={valid_error*100:.2f}%, norm={fg_norm:.5e}, {iteration}/{patience})"
#                     )

#                     if converged:
#                         info(f"Model converged.")
#                         if self.results.best_step != step:
#                             warning(
#                                 "Model converged before reaching maximum likelihood. Reduce base_learning_rate or increase batch_size."
#                             )
#                     else:
#                         info(
#                             f"Maximum number of iterations reached: {iteration}/{patience}"
#                         )

#                     # break loop if convergence reached
#                     done_looping = True
#                     break

#                 # increment iteration
#                 iteration += 1

#                 if self.config.batch_shuffle:
#                     # get index and shift slices from rng if batch_shuffle is True
#                     index = self.rng.integers(0, db.n_train_batches)
#                     shift = self.rng.integers(0, batch_size)

#                 # get data from dataset
#                 batch_data = db.train(x_and_y, index, batch_size, shift)

#                 # update model parameters with batch_data and learning rate
#                 self.update_wrt_cost(*batch_data, learning_rate)

#                 # validate model every validation_freq iteration
#                 if iteration % validation_freq != 0:
#                     continue

#                 # compute the loglikelihood at validation
#                 train_loglike = self.loglikelihood(*db.train(x_and_y))

#                 self.results.performance_graph[step] = {
#                     "log likelihood": np.round(train_loglike, 3),
#                 }

#                 # check if model has converged
#                 if not converged:
#                     # calculate the full gradient norm for convergence using 2-norm
#                     # full_gradient = (beta - beta_new)/learning_rate
#                     # norm(full_gradient)
#                     diff = []
#                     for n, beta in enumerate(self.betas):
#                         b = beta.get_value()
#                         d = betas_prev[n] - b
#                         diff.append(d)

#                     fg_norm = np.sqrt(np.sum(np.square(diff)))
#                     if fg_norm < convergence_threshold:
#                         converged = True
#                     else:
#                         # increase patience if model is not converged
#                         patience = min(
#                             max(patience, iteration * patience_inc), max_steps
#                         )

#                     # save betas into betas_prev for next iteration
#                     betas_prev = [b.get_value() for b in self.betas]

#                 # maximum likelihood check step
#                 if train_loglike < self.results.best_loglikelihood:
#                     # skip to next iteration if maximum likelihood not reached
#                     continue

#                 valid_error = self.prediction_error(*db.valid(x_and_y))
#                 if fg_norm < (fg_norm_tol / 10.0):
#                     fg_norm_tol = fg_norm
#                     info(
#                         f"Train (Step={step}, LL={train_loglike:.2f}, Error={valid_error*100:.2f}%, norm={fg_norm:.5e}, {iteration}/{patience})"
#                     )

#                 self.results.best_step = step
#                 self.results.best_iteration = iteration
#                 self.results.best_loglikelihood = train_loglike
#                 self.results.best_valid_error = valid_error
#                 self.results.betas = self.betas

#         train_time = round(perf_counter() - start_time, 3)
#         self.results.train_time = time_format(train_time)
#         self.results.iterations_per_sec = round(iteration / train_time, 2)
#         info(f"End (t={self.results.train_time})")
#         info(
#             f"Best results obtained at Step {self.results.best_step}: LL={self.results.best_loglikelihood:.2f}, Error={self.results.best_valid_error*100:.2f}%"
#         )

#         # save results
#         self.results.n_train = db.n_train
#         self.results.n_valid = db.n_valid
#         self.results.n_params = self.n_params
#         self.results.seed = self.config.seed
#         self.results.lr_history_graph = self.config.lr_scheduler.history
#         self.betas = self.results.betas

#         # Calculate the 1st and 2nd order gradients
#         n = db.n_train
#         k = len([p for p in self.results.betas if (p.status != 1)])
#         G = np.zeros((n, k))
#         H = np.zeros((n, k, k))
#         full_data = db.train(x_and_y, numpy_out=True)
#         for i in range(n):
#             g_vector = self.G(*[[d[i]] for d in full_data])
#             hess = self.H(*[[d[i]] for d in full_data])
#             G[i, :] = g_vector
#             H[i, :, :] = hess
#         self.results.hessian_matrix = H
#         self.results.bhhh_matrix = G[:, :, None] * G[:, None, :]
#         self.results.gnorm = fg_norm

#     def __str__(self):
#         return f"{self.name}"
