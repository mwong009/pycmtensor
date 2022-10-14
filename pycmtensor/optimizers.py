# optimizers.py
"""PyCMTensor optimizers module"""
import aesara
import aesara.tensor as aet

from .data import FLOATX


class Optimizer:
    def __init__(self, params, name, b1=0.0, b2=0.0, m=0.0, rho=0.0, epsilon=1e-8):
        """Base optimizer class

        Args:
            params (list): a list of :class:`expressions.TensorVariable` type objects.
                Used for constructing optimizer parameters.

        """
        self.name = name
        self._m = []
        self._v = []
        self._accu = []
        self._delta = []
        self._velocity = []
        for param in params:
            if param.status == 1:
                continue
            p = param()
            if (b1 > 0.0) and (b2 > 0.0):
                self._m.append(aesara.shared(aet.zeros_like(p).eval()))
                self._v.append(aesara.shared(aet.zeros_like(p).eval()))
            if rho > 0.0:
                self._accu.append(aesara.shared(aet.zeros_like(p).eval()))
                self._delta.append(aesara.shared(aet.zeros_like(p).eval()))
            if m > 0.0:
                self._velocity.append(aesara.shared(aet.zeros_like(p).eval()))

        self.epsilon = epsilon
        self._t = aesara.shared(0.0)
        self.b1 = b1
        self.b2 = b2
        self.rho = rho
        self.m = m

    def __repr__(self):
        return f"{self.name}"


class Adam(Optimizer):
    def __init__(self, params: list, b1=0.9, b2=0.999, **kwargs):
        """An optimizer that implments the Adam algorithm [#]_

        Args:
            params (list): a list of :class:`Betas` and/or :class:`Weights`
            b1 (float, optional): exponential decay rate for the 1st moment estimates.
                Defaults to ``0.9``
            b2 (float, optional): exponential decay rate for the 2nd moment estimates.
                Defaults to ``0.999``

        .. [#] Kingma et al., 2014. Adam: A Method for Stochastic Optimization. http://arxiv.org/abs/1412.6980
        """
        super().__init__(params, name="Adam", b1=b1, b2=b2)

    def update(self, cost, params: list, lr=0.001):
        """Caller to the optimizer class to generate a list of updates

        Args:
            cost (TensorVariable): a scalar element for the expression of the cost
                function where the derivatives are calculated
            params (list): a list of :class:`Betas` and/or :class:`Weights`
            lr (float, optional): learning rate. Defaults to 0.001

        Returns:
            list: a list of tuples of ``(p, p_t), (m, m_t), (v, v_t), (t, t_new)``
        """
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        t = self._t
        m_prev = self._m
        v_prev = self._v

        t_new = t + 1.0
        a_t = aet.sqrt(1.0 - self.b2**t_new) / (1.0 - self.b1**t_new)

        for m, v, param, grad in zip(m_prev, v_prev, params, grads):
            m_t = self.b1 * m + (1.0 - self.b1) * grad
            v_t = self.b2 * v + (1.0 - self.b2) * grad**2
            g_t = lr * a_t * m_t / (aet.sqrt(v_t) + self.epsilon)
            p_t = param - g_t
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((param, p_t))

        updates.append((t, t_new))

        return updates


class Adamax(Optimizer):
    def __init__(self, params: list, b1=0.9, b2=0.999, **kwargs):
        """An optimizer that implements the Adamax algorithm [#]_. It is a variant of
        the Adam algorithm

        Args:
            params (list): a list of :class:`Betas` and/or :class:`Weights`
            b1 (float, optional): exponential decay rate for the 1st moment estimates.
                Defaults to ``0.9``
            b2 (float, optional): exponential decay rate for the 2nd moment estimates.
                Defaults to ``0.999``

        .. [#] Kingma et al., 2014. Adam: A Method for Stochastic Optimization. http://arxiv.org/abs/1412.6980
        """
        super().__init__(params, name="Adamax", b1=b1, b2=b2)

    def update(self, cost, params: list, lr=0.001):
        """Caller to the optimizer class to generate a list of updates

        Args:
            cost (TensorVariable): a scalar element for the expression of the cost function where the derivatives are calculated
            params (list): a list of :class:`Betas` and/or :class:`Weights`
            lr (float, optional): learning rate. Defaults to ``0.001``

        Returns:
            list: a list of tuples of ``(p, p_t), (m, m_t), (v, v_t), (t, t_new)``
        """
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        t = self._t
        m_prev = self._m
        v_prev = self._v

        t_new = t + 1.0
        a_t = lr / (1.0 - self.b1**t)

        for m, v, param, grad in zip(m_prev, v_prev, params, grads):
            m_t = self.b1 * m + (1.0 - self.b1) * grad
            v_t = aet.maximum(self.b2 * v, abs(grad))
            g_t = a_t * m_t / (v_t + self.epsilon)
            p_t = param - g_t
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((param, p_t))

        updates.append((t, t_new))

        return updates


class Adadelta(Optimizer):
    def __init__(self, params: list, rho=0.95, **kwargs):
        """An optimizer that implements the Adadelta algorithm [#]_

        Adadelta is a stochastic gradient descent method that is based on adaptive
        learning rate per dimension to address two drawbacks:

        - The continual decay of learning rates throughout training
        - The need for a manually selected global learning rate

        Args:
            params (list): a list of :class:`Betas` and/or :class:`Weights`
            rho (float, optional): the decay rate for learning rate.
                Defaults to ``0.95``

        .. [#] Zeiler, 2012. ADADELTA: An Adaptive Learning Rate Method. http://arxiv.org/abs/1212.5701
        """
        super().__init__(params, name="Adadelta", rho=rho)

    def update(self, cost, params: list, lr=1.0):
        """Caller to the optimizer class to generate a list of updates

        Args:
            cost (TensorVariable): a scalar element for the expression of the cost function where the derivatives are calculated
            params (list): a list of :class:`Betas` and/or :class:`Weights`
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
        accumulator = self._accu
        delta = self._delta
        for a, d, param, grad in zip(accumulator, delta, params, grads):
            # update accumulator
            a_t = self.rho * a + (1.0 - self.rho) * grad**2
            # compute parameter update, using previous delta
            g_t = grad * aet.sqrt(d + self.epsilon) / aet.sqrt(a_t + self.epsilon)
            p_t = param - lr * g_t
            d_t = self.rho * d + (1.0 - self.rho) * g_t**2
            updates.append((param, p_t))
            updates.append((a, a_t))
            updates.append((d, d_t))

        return updates


class RMSProp(Optimizer):
    def __init__(self, params: list, rho=0.9, **kwargs):
        """An optimizer that implements the RMSprop algorithm [#]_

        Args:
            params (list): a list of :class:`Betas` and/or :class:`Weights`
            rho (float, optional): discounting factor for the history/coming gradient.
                Defaults to ``0.9``

        .. [#] Hinton, 2012. rmsprop: Divide the gradient by a running average of its recent magnitude. http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
        """
        super().__init__(params, name="RMSProp", rho=rho)

    def update(self, cost, params: list, lr=0.001):
        """Caller to the optimizer class to generate a list of updates

        Args:
            cost (TensorVariable): a scalar element for the expression of the cost function where the derivatives are calculated
            params (list): a list of :class:`Betas` and/or :class:`Weights`
            lr (float, optional): learning rate. Defaults to ``0.001``

        Returns:
            list: a list of tuples of ``(param, param_new), (a, a_t)``
        """
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        accumulator = self._accu
        for a, param, grad in zip(accumulator, params, grads):
            a_t = self.rho * a + (1.0 - self.rho) * grad**2
            g_t = lr / aet.sqrt(a_t + self.epsilon) * grad
            p_t = param - g_t

            updates.append((a, a_t))
            updates.append((param, p_t))

        return updates


class Momentum(Optimizer):
    def __init__(self, params: list, momentum=0.9, nesterov=True, **kwargs):
        """An optimizer that implements the Momentum algorithm [#]_

        Args:
            params (list): a list of :class:`Betas` and/or :class:`Weights`
            momentum (float, optional): acceleration factor in the relevant direction
                and dampens oscillations. Defaults to ``0.9``
            nesterov (bool, optional): whether to apply Nesterov momentum.
                Defaults to ``False``

        .. [#] Sutskever et al., 2013. On the importance of initialization and momentum in deep learning. http://jmlr.org/proceedings/papers/v28/sutskever13.pdf
        """
        super().__init__(params, name="Momentum", m=momentum)
        self.nesterov = nesterov
        if self.nesterov:
            self.name = "NAG"

    def update(self, cost, params: list, lr=0.001):
        """Caller to the optimizer class to generate a list of updates

        Args:
            cost (TensorVariable): a scalar element for the expression of the cost function where the derivatives are calculated
            params (list): a list of :class:`Betas` and/or :class:`Weights`
            lr (float, optional): the learning rate. Defaults to ``0.001``

        Returns:
            list: a list of tuples of ``(param, param_new), (v, v_t)``
        """
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        velocity = self._velocity
        for v, param, grad in zip(velocity, params, grads):
            v_t = self.m * v - lr * grad
            if self.nesterov:
                g_t = self.m * v_t - lr * grad
            else:
                g_t = v_t
            p_t = param + g_t

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
            params (list): a list of :class:`Betas` and/or :class:`Weights`

        .. [#] Duchi et al., 2011. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
        """
        super().__init__(params, name="AdaGrad", rho=0.1)

    def update(self, cost, params: list, lr=1.0):
        """Caller to the optimizer class to generate a list of updates

        Args:
            cost (TensorVariable): a scalar element for the expression of the cost function where the derivatives are calculated
            params (list): a list of :class:`Betas` and/or :class:`Weights`
            lr (float, optional): the learning rate. Defaults to ``1.0``

        Returns:
            list: a list of tuples of ``(param, param_new), (a, a_t)``
        """
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        accumulator = self._accu
        for param, grad, a in zip(params, grads, accumulator):
            a_t = a + grad**2
            g_t = lr / aet.sqrt(a_t + self.epsilon) * grad
            p_t = param - g_t
            updates.append((a, a_t))
            updates.append((param, p_t))

        return updates


class SGD(Optimizer):
    def __init__(self, params: list, **kwargs):
        """An optimizer that implements the stochastic gradient algorithm

        Args:
            params (list): a list of :class:`Betas` and/or :class:`Weights`
        """
        super().__init__(params, name="SGD")

    def update(self, cost, params: list, lr=0.001):
        """Caller to the optimizer class to generate a list of updates

        Args:
            cost (TensorVariable): a scalar element for the expression of the cost function where the derivatives are calculated
            params (list): a list of :class:`Betas` and/or :class:`Weights`
            lr (float, optional): the learning rate. Defaults to ``0.001``

        Returns:
            list: a list of tuples of ``(param, param_new)``
        """
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        for param, grad in zip(params, grads):
            p_t = param - lr * grad
            updates.append((param, p_t))

        return updates
