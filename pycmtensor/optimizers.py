# optimizers.py
"""PyCMTensor optimizers module"""
import aesara
import aesara.tensor as aet
from aesara import shared
from aesara.tensor.sharedvar import TensorSharedVariable

FLOATX = aesara.config.floatX

__all__ = [
    "Adam",
    "Nadam",
    "Adamax",
    "Adadelta",
    "RMSProp",
    "Momentum",
    "NAG",
    "AdaGrad",
    "SGD",
]


class Optimizer:
    def __init__(self, name, epsilon=1e-8, **kwargs):
        """Base optimizer class

        Args:
            name (str): name of the optimizer

        """
        self.name = name
        self._epsilon = epsilon

    @property
    def epsilon(self):
        return self._epsilon

    def __repr__(self):
        return f"{self.name}"


class Adam(Optimizer):
    def __init__(self, params: list, b1: float = 0.9, b2: float = 0.999, **kwargs):
        """An optimizer that implments the Adam algorithm [#]_

        Args:
            params (list): a list of ``TensorSharedVariable``
            b1 (float, optional): exponential decay rate for the 1st moment estimates.
                Defaults to ``0.9``
            b2 (float, optional): exponential decay rate for the 2nd moment estimates.
                Defaults to ``0.999``

        .. [#] Kingma et al., 2014. Adam: A Method for Stochastic Optimization. http://arxiv.org/abs/1412.6980
        """
        super().__init__(name="Adam")
        self.b1 = b1
        self.b2 = b2
        self._t = aesara.shared(1.0)
        self._m = [shared(aet.zeros_like(p()).eval()) for p in params if p.status != 1]
        self._v = [shared(aet.zeros_like(p()).eval()) for p in params if p.status != 1]

    @property
    def t(self):
        return self._t

    @property
    def m_prev(self):
        return self._m

    @property
    def v_prev(self):
        return self._v

    def update(self, cost, params: list, lr: float = 0.001):
        """Generate a list of updates

        Args:
            cost (TensorVariable): a scalar element for the expression of the cost
                function where the derivatives are calculated
            params (list): a list of ``TensorSharedVariable``
            lr (float, optional): learning rate. Defaults to 0.001

        Returns:
            list: a list of tuples of ``(p, p_t), (m, m_t), (v, v_t), (t, t_new)``
        """
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []

        t_new = self.t + 1.0
        for m, v, param, grad in zip(self.m_prev, self.v_prev, params, grads):
            m_t = self.b1 * m + (1.0 - self.b1) * grad
            m_t_hat = m_t / (1.0 - aet.pow(self.b1, self.t))

            v_t = self.b2 * v + (1.0 - self.b2) * aet.sqr(grad)
            v_t_hat = v_t / (1.0 - aet.pow(self.b2, self.t))

            g_t = lr * m_t_hat / (aet.sqrt(v_t_hat) + self.epsilon)
            p_t = param - g_t

            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((param, p_t))

        updates.append((self.t, t_new))

        return updates


class Nadam(Adam):
    def __init__(self, params: list, b1: float = 0.99, b2: float = 0.999, **kwargs):
        """An optimizer that implements the Nesterov Adam algorithm [#]_

        Args:
            params (list): a list of ``TensorSharedVariable``
            b1 (float, optional): exponential decay rate for the 1st moment estimates.
                Defaults to ``0.9``
            b2 (float, optional): exponential decay rate for the 2nd moment estimates.
                Defaults to ``0.999``

        .. [#] Dozat, T., 2016. Incorporating nesterov momentum into adam.(2016). Dostupné z: http://cs229.stanford.edu/proj2015/054_report.pdf.
        """
        super().__init__(params, b1, b2)
        self.name = "Nadam"

    def update(self, cost, params: list, lr: float = 0.001):
        """Generate a list of updates

        Args:
            cost (TensorVariable): a scalar element for the expression of the cost
                function where the derivatives are calculated
            params (list): a list of ``TensorSharedVariable``
            lr (float, optional): learning rate. Defaults to 0.001

        Returns:
            list: a list of tuples of ``(p, p_t), (m, m_t), (v, v_t), (t, t_new)``
        """
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []

        t_new = self.t + 1.0
        b1 = self.b1 * (1 - 0.5 * aet.pow(0.96, self.t / 250))
        b1_t = self.b1 * (1 - 0.5 * aet.pow(0.96, t_new / 250))
        for m, v, param, grad in zip(self.m_prev, self.v_prev, params, grads):
            g_t = grad / (1.0 - aet.pow(self.b1, self.t))

            m_t = self.b1 * m + (1.0 - self.b1) * grad
            m_t_hat = m_t / (1.0 - aet.pow(self.b1, t_new))

            v_t = self.b2 * v + (1.0 - self.b2) * aet.sqr(grad)
            v_t_hat = v_t / (1.0 - aet.pow(self.b2, self.t))

            m_t_hat = (1 - b1) * g_t + b1_t * m_t_hat
            g_t_hat = lr * m_t_hat / (aet.sqrt(v_t_hat) + self.epsilon)
            p_t = param - g_t_hat

            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((param, p_t))

        updates.append((self.t, t_new))

        return updates


class Adamax(Adam):
    def __init__(self, params: list, b1: float = 0.9, b2: float = 0.999, **kwargs):
        """An optimizer that implements the Adamax algorithm [#]_. It is a variant of
        the Adam algorithm

        Args:
            params (list): a list of ``TensorSharedVariable``
            b1 (float, optional): exponential decay rate for the 1st moment estimates.
                Defaults to ``0.9``
            b2 (float, optional): exponential decay rate for the 2nd moment estimates.
                Defaults to ``0.999``

        .. [#] Kingma et al., 2014. Adam: A Method for Stochastic Optimization. http://arxiv.org/abs/1412.6980
        """
        super().__init__(params, b1, b2)
        self.name = "Adamax"

    def update(self, cost, params: list, lr: float = 0.001):
        """Caller to the optimizer class to generate a list of updates

        Args:
            cost (TensorVariable): a scalar element for the expression of the cost function where the derivatives are calculated
            params (list): a list of ``TensorSharedVariable``
            lr (float, optional): learning rate. Defaults to ``0.001``

        Returns:
            list: a list of tuples of ``(p, p_t), (m, m_t), (v, v_t), (t, t_new)``
        """
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []

        t_new = self.t + 1.0
        a_t = lr / (1.0 - aet.pow(self.b1, self.t))

        for m, v, param, grad in zip(self.m_prev, self.v_prev, params, grads):
            m_t = self.b1 * m + (1.0 - self.b1) * grad
            v_t = aet.maximum(self.b2 * v, aet.abs(grad))
            g_t = a_t * m_t / (v_t + self.epsilon)
            p_t = param - g_t
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((param, p_t))

        updates.append((self.t, t_new))

        return updates


class Adadelta(Optimizer):
    def __init__(self, params: list, rho: float = 0.95, **kwargs):
        """An optimizer that implements the Adadelta algorithm [#]_

        Adadelta is a stochastic gradient descent method that is based on adaptive
        learning rate per dimension to address two drawbacks:

        - The continual decay of learning rates throughout training
        - The need for a manually selected global learning rate

        Args:
            params (list): a list of ``TensorSharedVariable``
            rho (float, optional): the decay rate for learning rate.
                Defaults to ``0.95``

        .. [#] Zeiler, 2012. ADADELTA: An Adaptive Learning Rate Method. http://arxiv.org/abs/1212.5701
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

    def update(self, cost, params: list, lr: float = 1.0):
        """Caller to the optimizer class to generate a list of updates

        Args:
            cost (TensorVariable): a scalar element for the expression of the cost function where the derivatives are calculated
            params (list): a list of ``TensorSharedVariable``
            lr (float, optional): learning rate. Defaults to ``1.0``

        Returns:
            list: a list of tuples of ``(param, param_new), (a, a_t), (d, d_t)``

        .. Note::

            Since the Adadelta algorithm uses an adaptive learning rate, the
            learning rate is set to ``1.0``
        """
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        for accu, d, param, grad in zip(self.accumulator, self.delta, params, grads):
            # update accumulator
            accu_t = self.rho * accu + (1.0 - self.rho) * grad**2
            # compute parameter update, using previous delta
            g_t = grad * aet.sqrt(d + self.epsilon) / aet.sqrt(accu_t + self.epsilon)
            p_t = param - lr * g_t
            d_t = self.rho * d + (1.0 - self.rho) * g_t**2
            updates.append((param, p_t))
            updates.append((accu, accu_t))
            updates.append((d, d_t))

        return updates


class RMSProp(Optimizer):
    def __init__(self, params: list, rho: float = 0.9, **kwargs):
        """An optimizer that implements the RMSprop algorithm [#]_

        Args:
            params (list): a list of ``TensorSharedVariable``
            rho (float, optional): discounting factor for the history/coming gradient.
                Defaults to ``0.9``

        .. [#] Hinton, 2012. rmsprop: Divide the gradient by a running average of its recent magnitude. http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
        """
        super().__init__(name="RMSProp")
        self.rho = rho
        self._accu = [
            shared(aet.zeros_like(p()).eval()) for p in params if p.status != 1
        ]

    @property
    def accumulator(self):
        return self._accu

    def update(self, cost, params: list, lr: float = 0.001):
        """Caller to the optimizer class to generate a list of updates

        Args:
            cost (TensorVariable): a scalar element for the expression of the cost function where the derivatives are calculated
            params (list): a list of ``TensorSharedVariable``
            lr (float, optional): learning rate. Defaults to ``0.001``

        Returns:
            list: a list of tuples of ``(param, param_new), (a, a_t)``
        """
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        for accu, param, grad in zip(self.accumulator, params, grads):
            accu_t = self.rho * accu + (1.0 - self.rho) * aet.sqr(grad)
            g_t = lr / aet.sqrt(accu_t + self.epsilon) * grad
            p_t = param - g_t

            updates.append((accu, accu_t))
            updates.append((param, p_t))

        return updates


class Momentum(Optimizer):
    def __init__(self, params: list, mu: float = 0.9, **kwargs):
        """An optimizer that implements the Momentum algorithm [#]_

        Args:
            params (list): a list of ``TensorSharedVariable``
            mu (float, optional): acceleration factor in the relevant direction
                and dampens oscillations. Defaults to ``0.9``

        .. [#] Sutskever et al., 2013. On the importance of initialization and momentum in deep learning. http://jmlr.org/proceedings/papers/v28/sutskever13.pdf
        """
        super().__init__(name="Momentum")
        self.mu = mu
        self._v = [shared(aet.zeros_like(p()).eval()) for p in params if p.status != 1]

    @property
    def velocity(self):
        return self._v

    def update(self, cost, params: list, lr: float = 0.001):
        """Caller to the optimizer class to generate a list of updates

        Args:
            cost (TensorVariable): a scalar element for the expression of the cost function where the derivatives are calculated
            params (list): a list of ``TensorSharedVariable``
            lr (float, optional): the learning rate. Defaults to ``0.001``

        Returns:
            list: a list of tuples of ``(param, param_new), (v, v_t)``
        """
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        for v, param, grad in zip(self.velocity, params, grads):
            v_t = self.mu * v + grad
            p_t = param - lr * v_t

            updates.append((v, v_t))
            updates.append((param, p_t))

        return updates


class NAG(Momentum):
    def __init__(self, params: list, mu: float = 0.99, **kwargs):
        """An optimizer that implements the Nestrov Accelerated Gradient algorithm [#]_

        Args:
            params (list): a list of ``TensorSharedVariable``
            mu (float, optional): acceleration factor in the relevant direction
                and dampens oscillations. Defaults to ``0.9``

        .. [#] Sutskever et al., 2013. On the importance of initialization and momentum in deep learning. http://jmlr.org/proceedings/papers/v28/sutskever13.pdf
        """
        super().__init__(params, mu)
        self.name = "NAG"
        self._t = aesara.shared(0.0)

    @property
    def t(self):
        return self._t

    def update(self, cost, params: list, lr: float = 0.001):
        """Caller to the optimizer class to generate a list of updates

        Args:
            cost (TensorVariable): a scalar element for the expression of the cost function where the derivatives are calculated
            params (list): a list of ``TensorSharedVariable``
            lr (float, optional): the learning rate. Defaults to ``0.001``

        Returns:
            list: a list of tuples of ``(param, param_new), (v, v_t)``
        """
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        t_new = self.t + 1.0
        mu = self.mu * (1 - 0.5 * aet.pow(0.96, self.t / 250))
        mu_t = self.mu * (1 - 0.5 * aet.pow(0.96, t_new / 250))
        for v, param, grad in zip(self.velocity, params, grads):
            v_t = mu * v + grad
            v_t_hat = grad + mu_t * v_t
            p_t = param - lr * v_t_hat

            updates.append((v, v_t))
            updates.append((param, p_t))

        return updates


class AdaGrad(Optimizer):
    def __init__(self, params: list, **kwargs):
        """An optimizer that implements the Adagrad algorithm [#]_

        Adagrad is an optimizer with parameter-specific learning rates, which are
        adapted relative to how frequently a parameter gets updated during training.
        The more updates a parameter receives, the smaller the updates.

        Args:
            params (list): a list of ``TensorSharedVariable``

        .. [#] Duchi et al., 2011. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
        """
        super().__init__(name="AdaGrad")
        self._accu = [
            shared(aet.zeros_like(p()).eval()) for p in params if p.status != 1
        ]

    @property
    def accumulator(self):
        return self._accu

    def update(self, cost, params: list, lr: float = 1.0):
        """Caller to the optimizer class to generate a list of updates

        Args:
            cost (TensorVariable): a scalar element for the expression of the cost function where the derivatives are calculated
            params (list): a list of ``TensorSharedVariable``
            lr (float, optional): the learning rate. Defaults to ``1.0``

        Returns:
            list: a list of tuples of ``(param, param_new), (accu, accu_t)``
        """
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        for param, grad, accu in zip(params, grads, self.accumulator):
            accu_t = accu + aet.sqr(grad)
            g_t = lr / aet.sqrt(accu_t + self.epsilon) * grad
            p_t = param - g_t
            updates.append((accu, accu_t))
            updates.append((param, p_t))

        return updates


class SGD(Optimizer):
    def __init__(self, params: list, **kwargs):
        """An optimizer that implements the stochastic gradient algorithm

        Args:
            params (list): a list of ``TensorSharedVariable``
        """
        super().__init__(name="SGD")

    def update(self, cost, params: list, lr: float = 0.001):
        """Caller to the optimizer class to generate a list of updates

        Args:
            cost (TensorVariable): a scalar element for the expression of the cost function where the derivatives are calculated
            params (list): a list of ``TensorSharedVariable``
            lr (float, optional): the learning rate. Defaults to ``0.001``

        Returns:
            list: a list of ``(param, param_new)`` tuples
        """
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        for param, grad in zip(params, grads):
            p_t = param - lr * grad
            updates.append((param, p_t))

        return updates
