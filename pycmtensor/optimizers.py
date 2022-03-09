# optimizers.py

import aesara
import aesara.tensor as aet
import numpy as np

floatX = aesara.config.floatX


class Optimizer:
    def __init__(self, params, name, b1=0.0, b2=0.0, m=0.0, rho=0.0, epsilon=1e-8):
        self.name = name
        params = [p() for p in params if p.status != 1]
        zero = np.array(0.0).astype(floatX)
        self.epsilon = epsilon
        if (b1 > 0.0) and (b2 > 0.0):
            self._m = [aesara.shared(p.get_value() * zero) for p in params]
            self._v = [aesara.shared(p.get_value() * zero) for p in params]
            self._t = aesara.shared(zero)
            self.b1 = b1
            self.b2 = b2
        if rho > 0.0:
            self._accu = [aesara.shared(p.get_value() * zero) for p in params]
            self._delta = [aesara.shared(p.get_value() * zero) for p in params]
            self.rho = rho
        if m > 0.0:
            self._velocity = [aesara.shared(p.get_value() * zero) for p in params]
            self.m = m

    def __repr__(self):
        return f"{self.name}"


class Adam(Optimizer):
    def __init__(self, params, b1=0.9, b2=0.999):
        super().__init__(params, name="Adam", b1=b1, b2=b2)

    def update(self, cost, params, lr=0.001):
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
    def __init__(self, params, b1=0.9, b2=0.999):
        super().__init__(params, name="Adamax", b1=b1, b2=b2)

    def update(self, cost, params, lr=0.001):
        params = [p() for p in params if p.status != 1]
        one = np.array(1.0).astype(floatX)
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        t = self._t
        m_prev = self._m
        v_prev = self._v

        t_new = t + 1.0
        a_t = lr / (one - self.b1**t)

        for m, v, param, grad in zip(m_prev, v_prev, params, grads):
            m_t = self.b1 * m + (one - self.b1) * grad
            v_t = aet.maximum(self.b2 * v, abs(grad))
            g_t = a_t * m_t / (v_t + self.epsilon)
            p_t = param - g_t
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((param, p_t))

        updates.append((t, t_new))

        return updates


class Adadelta(Optimizer):
    def __init__(self, params, rho=0.95):
        super().__init__(params, name="Adadelta", rho=rho)

    def update(self, cost, params, lr=1.0):
        params = [p() for p in params if p.status != 1]
        one = np.array(1.0).astype(floatX)
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        accumulator = self._accu
        delta = self._delta
        for a, d, param, grad in zip(accumulator, delta, params, grads):
            # update accumulator
            a_t = self.rho * a + (one - self.rho) * grad**2
            # compute parameter update, using previous delta
            g_t = grad * aet.sqrt(d + self.epsilon) / aet.sqrt(a_t + self.epsilon)
            p_t = param - lr * g_t
            d_t = self.rho * d + (one - self.rho) * g_t**2
            updates.append((param, p_t))
            updates.append((a, a_t))
            updates.append((d, d_t))

        return updates


class RMSProp(Optimizer):
    def __init__(self, params, rho=0.9):
        super().__init__(params, name="RMSProp", rho=rho)

    def update(self, cost, params, lr=0.001):
        params = [p() for p in params if p.status != 1]
        one = np.array(1.0).astype(floatX)
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        accumulator = self._accu
        for a, param, grad in zip(accumulator, params, grads):
            a_t = self.rho * a + (one - self.rho) * grad**2
            g_t = lr / aet.sqrt(a_t + self.epsilon) * grad
            p_t = param - g_t

            updates.append((a, a_t))
            updates.append((param, p_t))

        return updates


class MomentumSGD(Optimizer):
    def __init__(self, params, momentum=0.9, nesterov=False):
        super().__init__(params, name="MomentumSGD", m=momentum)
        self.nesterov = nesterov
        if self.nesterov:
            self.name = "Nesterov"

    def update(self, cost, params, lr=0.001):
        params = [p() for p in params if p.status != 1]

        updates = []
        velocity = self._velocity
        if self.nesterov:
            for n, (v, p) in enumerate(zip(velocity, params)):
                a_t = p - self.m * v
                params[n] = a_t

        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        for v, param, grad in zip(velocity, params, grads):
            v_t = self.m * v + lr * grad
            p_t = param - v_t

            updates.append((v, v_t))
            updates.append((param, p_t))

        return updates


class AdaGrad(Optimizer):
    def __init__(self, name="SGD", params=None):
        super().__init__(params, name)

    def update(self, cost, params, lr=0.01):
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
    def __init__(self, name="SGD", params=None):
        super().__init__(params, name)

    def update(self, cost, params, lr=0.001):
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        for param, grad in zip(params, grads):
            p_t = param - lr * grad
            updates.append((param, p_t))

        return updates
