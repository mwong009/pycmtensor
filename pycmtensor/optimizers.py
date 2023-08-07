# optimizers.py
"""PyCMTensor optimizers module"""
import aesara
import aesara.tensor as aet
from aesara import shared
from aesara.ifelse import ifelse

import pycmtensor.defaultconfig as defaultconfig

config = defaultconfig.config

FLOATX = aesara.config.floatX

__all__ = [
    "Adam",
    "Nadam",
    "Adamax",
    "Adadelta",
    "RMSProp",
    "RProp",
    "Momentum",
    "NAG",
    "AdaGrad",
    "SGD",
    "SQNBFGS",
]


class Optimizer:
    def __init__(self, name, epsilon=1e-8, **kwargs):
        """Base optimizer class

        Args:
            name (str): name of the optimizer

        """
        self.name = name
        self._epsilon = shared(epsilon, name="epsilion")

    @property
    def epsilon(self):
        return self._epsilon

    def __repr__(self):
        return f"{self.name}"

    def update(self, cost, params, lr):
        """Update parameters for aesara function calls

        Args:
            cost (TensorVariable): a scalar element for the expression of the cost
                function where the derivatives are calculated
            params (list[TensorSharedVariable]): parameters of the model
            lr (Union[float, TensorSharedVariable]): the learning rate

        Returns:
            (list): a list of `(param, param_new)` tuple pairs
        """
        pass


class Adam(Optimizer):
    def __init__(self, params, b1=0.9, b2=0.999, **kwargs):
        """An optimizer that implments the Adam algorithm[^1]

        Args:
            params (list[TensorSharedVariable]): parameters of the model
            b1 (float, optional): exponential decay rate for the 1st moment estimates.
                Defaults to `0.9`
            b2 (float, optional): exponential decay rate for the 2nd moment estimates.
                Defaults to `0.999`

        Attributes:
            t (TensorSharedVariable): time step
            m_prev (list[TensorSharedVariable]): previous time step momentum
            v_prev (list[TensorSharedVariable]): previous time step velocity

        [^1]: Kingma et al., 2014. Adam: A Method for Stochastic Optimization. http://arxiv.org/abs/1412.6980
        """
        super().__init__(name="Adam")
        self.b1 = aesara.shared(b1, name="b1")
        self.b2 = aesara.shared(b2, name="b2")
        self._t = aesara.shared(1.0, name="t")
        self._m = [
            shared(aet.zeros_like(p()).eval(), name=f"m_{p().name}")
            for p in params
            if p.status != 1
        ]
        self._v = [
            shared(aet.zeros_like(p()).eval(), name=f"v_{p().name}")
            for p in params
            if p.status != 1
        ]

    @property
    def t(self):
        return self._t

    @property
    def m_prev(self):
        return self._m

    @property
    def v_prev(self):
        return self._v

    def update(self, cost, params, lr):
        bounds = [(p.lb, p.ub) for p in params if p.status != 1]
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []

        t_new = self.t + 1.0
        for m, v, param, grad, b in zip(
            self.m_prev, self.v_prev, params, grads, bounds
        ):
            m_t = self.b1 * m + (1.0 - self.b1) * grad
            m_t_hat = m_t / (1.0 - aet.pow(self.b1, self.t))

            v_t = self.b2 * v + (1.0 - self.b2) * aet.sqr(grad)
            v_t_hat = v_t / (1.0 - aet.pow(self.b2, self.t))

            g_t = lr * m_t_hat / (aet.sqrt(v_t_hat) + self.epsilon)
            p_t = param - g_t
            p_t = clip(p_t, *b)

            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((param, p_t))

        updates.append((self.t, t_new))

        return updates


class Nadam(Adam):
    def __init__(self, params, b1=0.99, b2=0.999, **kwargs):
        """An optimizer that implements the Nesterov Adam algorithm[^1]

        Args:
            params (list[TensorSharedVariable]): parameters of the model
            b1 (float, optional): exponential decay rate for the 1st moment estimates.
                Defaults to `0.9`
            b2 (float, optional): exponential decay rate for the 2nd moment estimates.
                Defaults to `0.999`

        Attributes:
            t (TensorSharedVariable): time step
            m_prev (list[TensorSharedVariable]): previous time step momentum
            v_prev (list[TensorSharedVariable]): previous time step velocity

        [^1]: Dozat, T., 2016. Incorporating nesterov momentum into adam.(2016). Dostupné z: http://cs229.stanford.edu/proj2015/054_report.pdf.
        """
        super().__init__(params, b1, b2)
        self.name = "Nadam"

    def update(self, cost, params, lr):
        bounds = [(p.lb, p.ub) for p in params if p.status != 1]
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []

        t_new = self.t + 1.0
        b1 = self.b1 * (1 - 0.5 * aet.pow(0.96, self.t / 250))
        b1_t = self.b1 * (1 - 0.5 * aet.pow(0.96, t_new / 250))
        for m, v, param, grad, b in zip(
            self.m_prev, self.v_prev, params, grads, bounds
        ):
            g_t = grad / (1.0 - aet.pow(self.b1, self.t))

            m_t = self.b1 * m + (1.0 - self.b1) * grad
            m_t_hat = m_t / (1.0 - aet.pow(self.b1, t_new))

            v_t = self.b2 * v + (1.0 - self.b2) * aet.sqr(grad)
            v_t_hat = v_t / (1.0 - aet.pow(self.b2, self.t))

            m_t_hat = (1 - b1) * g_t + b1_t * m_t_hat
            g_t_hat = lr * m_t_hat / (aet.sqrt(v_t_hat) + self.epsilon)
            p_t = param - g_t_hat
            p_t = clip(p_t, *b)

            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((param, p_t))

        updates.append((self.t, t_new))

        return updates


class Adamax(Adam):
    def __init__(self, params, b1=0.9, b2=0.999, **kwargs):
        """An optimizer that implements the Adamax algorithm[^1]. It is a variant of
        the Adam algorithm

        Args:
            params (list[TensorSharedVariable]): parameters of the model
            b1 (float, optional): exponential decay rate for the 1st moment estimates.
                Defaults to `0.9`
            b2 (float, optional): exponential decay rate for the 2nd moment estimates.
                Defaults to `0.999`

        Attributes:
            t (TensorSharedVariable): time step
            m_prev (list[TensorSharedVariable]): previous time step momentum
            v_prev (list[TensorSharedVariable]): previous time step velocity

        [^1]: Kingma et al., 2014. Adam: A Method for Stochastic Optimization. http://arxiv.org/abs/1412.6980
        """
        super().__init__(params, b1, b2)
        self.name = "Adamax"

    def update(self, cost, params, lr):
        bounds = [(p.lb, p.ub) for p in params if p.status != 1]
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []

        t_new = self.t + 1.0
        a_t = lr / (1.0 - aet.pow(self.b1, self.t))

        for m, v, param, grad, b in zip(
            self.m_prev, self.v_prev, params, grads, bounds
        ):
            m_t = self.b1 * m + (1.0 - self.b1) * grad
            v_t = aet.maximum(self.b2 * v, aet.abs(grad))
            g_t = a_t * m_t / (v_t + self.epsilon)
            p_t = param - g_t
            p_t = clip(p_t, *b)

            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((param, p_t))

        updates.append((self.t, t_new))

        return updates


class Adadelta(Optimizer):
    def __init__(self, params, rho=0.95, **kwargs):
        """An optimizer that implements the Adadelta algorithm[^1]

        Adadelta is a stochastic gradient descent method that is based on adaptive
        learning rate per dimension to address two drawbacks:

        - The continual decay of learning rates throughout training
        - The need for a manually selected global learning rate

        Args:
            params (list[TensorSharedVariable]): parameters of the model
            rho (float, optional): the decay rate for learning rate.
                Defaults to `0.95`

        Attributes:
            accumulator (list[TensorSharedVariable]): gradient accumulator
            delta (list[TensorSharedVariable]): adaptive difference between gradients

        [^1]: Zeiler, 2012. ADADELTA: An Adaptive Learning Rate Method. http://arxiv.org/abs/1212.5701
        """
        super().__init__(name="Adadelta")
        self.rho = rho
        self._accu = [
            shared(aet.zeros_like(p()).eval()) for p in params if p.status != 1
        ]
        self._delta = [
            shared(aet.zeros_like(p()).eval()) for p in params if p.status != 1
        ]

    @property
    def accumulator(self):
        return self._accu

    @property
    def delta(self):
        return self._delta

    def update(self, cost, params, lr):
        bounds = [(p.lb, p.ub) for p in params if p.status != 1]
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        for accu, d, param, grad, b in zip(
            self.accumulator, self.delta, params, grads, bounds
        ):
            # update accumulator
            accu_t = self.rho * accu + (1.0 - self.rho) * grad**2
            # compute parameter update, using previous delta
            g_t = grad * aet.sqrt(d + self.epsilon) / aet.sqrt(accu_t + self.epsilon)
            d_t = self.rho * d + (1.0 - self.rho) * g_t**2
            p_t = param - lr * g_t
            p_t = clip(p_t, *b)

            updates.append((param, p_t))
            updates.append((accu, accu_t))
            updates.append((d, d_t))

        return updates


class RProp(Optimizer):
    def __init__(self, params, inc=1.05, dec=0.5, bounds=[1e-6, 50.0], **kwargs):
        """An optimizer that implements the Rprop algorithm[^1]

        Args:
            params (list[TensorSharedVariable]): parameters of the model
            inc (float, optional): increment step if same gradient direction
            dec (float, optional): decrement step if different gradient direction
            bounds (List[float]): min and maximum bounds for increment step

        Attributes:
            factor (List[TensorVariable]): learning rate factor multiplier (init=1.)
            ghat (List[TensorVariable]): previous step gradients

        [^1]: Igel, C., & Hüsken, M. (2003). Empirical evaluation of the improved Rprop learning algorithms. Neurocomputing, 50, 105-123.
        """
        super().__init__(name="RProp")
        self.inc = inc
        self.dec = dec
        self.bounds = bounds

        self._ghat = [
            shared(aet.zeros_like(p()).eval()) for p in params if p.status != 1
        ]
        self._factor = [
            shared(aet.ones_like(p()).eval()) for p in params if p.status != 1
        ]

    def update(self, cost, params, lr):
        bounds = [(p.lb, p.ub) for p in params if p.status != 1]
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        for param, grad, gh, f in zip(params, grads, self._ghat, self._factor):
            if aet.gt(grad * gh, 0.0):
                f_new = aet.clip(f * self.inc, *self.bounds)
            elif aet.lt(grad * gh, 0.0):
                f_new = aet.clip(f * self.dec, *self.bounds)
            else:
                f_new = f
            p_t = param - lr * f_new * grad

            updates.append((gh, grad))
            updates.append((param, p_t))
            updates.append((f, f_new))

        return updates


class RMSProp(Optimizer):
    def __init__(self, params, rho=0.9, **kwargs):
        """An optimizer that implements the RMSprop algorithm[^1]

        Args:
            params (list[TensorSharedVariable]): parameters of the model
            rho (float, optional): discounting factor for the history/coming gradient.
                Defaults to `0.9`

        Attributes:
            accumulator (TensorVariable): gradient accumulator

        [^1]: Hinton, 2012. rmsprop: Divide the gradient by a running average of its recent magnitude. http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
        """
        super().__init__(name="RMSProp")
        self.rho = shared(rho)
        self._accu = [
            shared(aet.zeros_like(p()).eval()) for p in params if p.status != 1
        ]

    @property
    def accumulator(self):
        return self._accu

    def update(self, cost, params, lr):
        bounds = [(p.lb, p.ub) for p in params if p.status != 1]
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        for accu, param, grad, b in zip(self.accumulator, params, grads, bounds):
            accu_t = self.rho * accu + (1.0 - self.rho) * aet.sqr(grad)
            g_t = lr / aet.sqrt(accu_t + self.epsilon) * grad
            p_t = param - g_t
            p_t = clip(p_t, *b)

            updates.append((accu, accu_t))
            updates.append((param, p_t))

        return updates


class Momentum(Optimizer):
    def __init__(self, params, mu=0.9, **kwargs):
        """An optimizer that implements the Momentum algorithm[^1]

        Args:
            params (list[TensorSharedVariable]): parameters of the model
            mu (float, optional): acceleration factor in the relevant direction
                and dampens oscillations. Defaults to `0.9`

        Attributes:
            velocity (list[TensorSharedVariable]): momentum velocity

        [^1]: Sutskever et al., 2013. On the importance of initialization and momentum in deep learning. http://jmlr.org/proceedings/papers/v28/sutskever13.pdf
        """
        super().__init__(name="Momentum")
        self.mu = shared(mu)
        self._v = [shared(aet.zeros_like(p()).eval()) for p in params if p.status != 1]

    @property
    def velocity(self):
        return self._v

    def update(self, cost, params, lr):
        bounds = [(p.lb, p.ub) for p in params if p.status != 1]
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        for v, param, grad, b in zip(self.velocity, params, grads, bounds):
            v_t = self.mu * v + grad
            p_t = param - lr * v_t
            p_t = clip(p_t, *b)

            updates.append((v, v_t))
            updates.append((param, p_t))

        return updates


class NAG(Momentum):
    def __init__(self, params, mu=0.99, **kwargs):
        """An optimizer that implements the Nestrov Accelerated Gradient algorithm[^1]

        Args:
            params (list[TensorSharedVariable]): parameters of the model
            mu (float, optional): acceleration factor in the relevant direction
                and dampens oscillations. Defaults to `0.9`

        Attributes:
            t (TensorSharedVariable): momentum time step
            velocity (list[TensorSharedVariable]): momentum velocity

        [^1]: Sutskever et al., 2013. On the importance of initialization and momentum in deep learning. http://jmlr.org/proceedings/papers/v28/sutskever13.pdf
        """
        super().__init__(params, mu)
        self.name = "NAG"
        self._t = shared(0.0)

    @property
    def t(self):
        return self._t

    def update(self, cost, params, lr):
        bounds = [(p.lb, p.ub) for p in params if p.status != 1]
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        t_new = self.t + 1.0
        mu = self.mu * (1 - 0.5 * aet.pow(0.96, self.t / 250))
        mu_t = self.mu * (1 - 0.5 * aet.pow(0.96, t_new / 250))
        for v, param, grad, b in zip(self.velocity, params, grads, bounds):
            v_t = mu * v + grad
            v_t_hat = grad + mu_t * v_t
            p_t = param - lr * v_t_hat
            p_t = clip(p_t, *b)

            updates.append((v, v_t))
            updates.append((param, p_t))

        return updates


class AdaGrad(Optimizer):
    def __init__(self, params, **kwargs):
        """An optimizer that implements the Adagrad algorithm[^1]

        Adagrad is an optimizer with parameter-specific learning rates, which are
        adapted relative to how frequently a parameter gets updated during training.
        The more updates a parameter receives, the smaller the updates.

        Args:
            params (list[TensorSharedVariable]): parameters of the model

        Attributes:
            accumulator (list[TensorSharedVariable]): gradient accumulators

        [^1]: Duchi et al., 2011. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
        """
        super().__init__(name="AdaGrad")
        self._accu = [
            shared(aet.zeros_like(p()).eval()) for p in params if p.status != 1
        ]

    @property
    def accumulator(self):
        return self._accu

    def update(self, cost, params, lr):
        bounds = [(p.lb, p.ub) for p in params if p.status != 1]
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        for param, grad, accu, b in zip(params, grads, self.accumulator, bounds):
            accu_t = accu + aet.sqr(grad)
            g_t = lr / aet.sqrt(accu_t + self.epsilon) * grad
            p_t = param - g_t
            p_t = clip(p_t, *b)
            updates.append((accu, accu_t))
            updates.append((param, p_t))

        return updates


class SGD(Optimizer):
    def __init__(self, params, **kwargs):
        """An optimizer that implements the stochastic gradient algorithm

        Args:
            params (list[TensorSharedVariable]): parameters of the model
        """
        super().__init__(name="SGD")

    def update(self, cost, params, lr):
        bounds = [(p.lb, p.ub) for p in params if p.status != 1]
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        for param, grad, b in zip(params, grads, bounds):
            p_t = param - lr * grad
            p_t = clip(p_t, *b)

            updates.append((param, p_t))

        return updates


class SQNBFGS(Optimizer):
    def __init__(self, params, config=None, **kwargs):
        """A L-BFGS optimizer implementing the adaptive stochastic Quasi-Newton (SQN) based approach [^1]

        Args:
            params (list[TensorSharedVariable]): parameters of the model
            config (pycmtensor.config): pycmtensor config object

        [^1]: Byrd, R. H., Hansen, S. L., Nocedal, J., & Singer, Y. (2016). A stochastic quasi-Newton method for large-scale optimization. SIAM Journal on Optimization, 26(2), 1008-1031.
        """
        super().__init__(name="BFGS")
        if config is not None:
            self.warmup = config.BFGS_warmup
        else:
            self.warmup = 20

        self._t = aesara.shared(1.0)
        self._y = [
            shared(aet.zeros_like(p()).eval()) for p in params if (p.status != 1)
        ]
        self._accu = [
            shared(aet.zeros_like(p()).eval()) for p in params if p.status != 1
        ]

        self._s = [shared(p().eval()) for p in params if (p.status != 1)]
        self._yhat = [shared(p().eval()) for p in params if (p.status != 1)]

        self._H0 = aesara.shared(aet.eye(len(self._y), dtype=FLOATX).eval())
        self.I = aesara.shared(aet.eye(len(self._y), dtype=FLOATX).eval())

    def update(self, cost, params, lr):
        T = self.warmup
        bounds = [(p.lb, p.ub) for p in params if p.status != 1]
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        for n, (param, grad, s, y, yh, a, b) in enumerate(
            zip(params, grads, self._s, self._y, self._yhat, self._accu, bounds)
        ):
            # perform warm up for a few runs
            B = ifelse(
                aet.lt(self._t, 2 * T),
                then_branch=grad,
                else_branch=aet.sum(self._H0[n, n] * grad),
            )
            # B = aet.sum(self._H0[n, n] * grad)

            a_t = a + aet.sqr(B)
            updates.append((a, a_t))

            p_t = param - lr / aet.sqrt(a_t + self.epsilon) * B
            p_t = clip(p_t, *b)
            updates.append((param, p_t))

            s_new = p_t - param
            updates.append((s, s_new))

            yh_t = grad
            updates.append((yh, yh_t))

            y_new = grad - yh
            updates.append((y, y_new))

        # BFGS update algorithm
        s_t = aet.atleast_2d(self._s, left=False)
        y_t = aet.atleast_2d(self._y, left=False)
        rho_t = 1 / (s_t.T @ y_t)
        li = self.I - rho_t * (s_t @ y_t.T)
        ri = self.I - rho_t * (y_t @ s_t.T)
        h_res = rho_t * (s_t @ s_t.T)

        # update hessian only every T steps to save computational time
        H_new = ifelse(
            aet.ge(self._t, T) * aet.eq(self._t % T, 0),
            then_branch=li @ self._H0 @ ri + h_res,
            else_branch=self._H0,
        )
        # H_new = li @ self._H0 @ ri + h_res
        updates.append((self._H0, H_new))

        updates.append((self._t, self._t + 1))

        return updates


def clip(param, min, max):
    if any([min, max]) and (config.beta_clipping):
        if min is None:
            min = -9999.0
        if max is None:
            max = 9999.0
        return aet.clip(param, min, max)
    else:
        return param
