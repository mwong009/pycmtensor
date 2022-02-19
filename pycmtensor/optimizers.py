# optimizers.py

import aesara
import aesara.tensor as aet
import numpy as np

floatX = aesara.config.floatX


class Adam:
    def __init__(self, params, b1=0.9, b2=0.999, epsilon=1e-8):
        params = [p() for p in params if p.status != 1]
        zero = np.array(0.0).astype(floatX)
        self._m = [aesara.shared(p.get_value() * zero) for p in params]
        self._v = [aesara.shared(p.get_value() * zero) for p in params]
        self._t = aesara.shared(zero)
        self.b1 = b1
        self.b2 = b2
        self.epsilon = epsilon

    def update(self, cost, params, lr=0.001):
        params = [p() for p in params if p.status != 1]
        one = np.array(1.0).astype(floatX)
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        t = self._t
        m_prev = self._m
        v_prev = self._v

        t_new = t + 1.0
        a_t = aet.sqrt(one - self.b2**t_new) / (one - self.b1**t_new)

        for m, v, param, grad in zip(m_prev, v_prev, params, grads):
            m_t = self.b1 * m + (one - self.b1) * grad
            v_t = self.b2 * v + (one - self.b2) * grad**2
            g_t = a_t * m_t / (aet.sqrt(v_t) + self.epsilon)
            p_t = param - lr * g_t
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((param, p_t))

        updates.append((t, t_new))

        return updates


class Adamax:
    def __init__(self, params, b1=0.9, b2=0.999, epsilon=1e-8):
        params = [p() for p in params if p.status != 1]
        zero = np.array(0.0).astype(floatX)
        self._m = [aesara.shared(p.get_value() * zero) for p in params]
        self._v = [aesara.shared(p.get_value() * zero) for p in params]
        self._t = aesara.shared(zero)
        self.b1 = b1
        self.b2 = b2
        self.epsilon = epsilon

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


class Adadelta:
    def __init__(self, params, rho=0.95, epsilon=1e-8):
        params = [p() for p in params if p.status != 1]
        zero = np.array(0.0).astype(floatX)
        self._accu = [aesara.shared(p.get_value() * zero) for p in params]
        self._delta = [aesara.shared(p.get_value() * zero) for p in params]
        self.rho = rho
        self.epsilon = epsilon

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


class RMSProp:
    def __init__(self, params, rho=0.9, epsilon=1e-8, decay=0.0):
        params = [p() for p in params if p.status != 1]
        zero = np.array(0.0).astype(floatX)
        self._a = [aesara.shared(p.get_value() * zero) for p in params]
        self.rho = rho
        self.epsilon = epsilon
        self.decay = decay

    def update(self, cost, params, lr=0.001):
        params = [p() for p in params if p.status != 1]
        one = np.array(1.0).astype(floatX)
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        accumulators = self._a
        for a, param, grad in zip(accumulators, params, grads):
            a_t = self.rho * a + (one - self.rho) * grad**2
            g_t = grad / (aet.sqrt(a_t) + self.epsilon)
            p_t = param - lr * g_t

            updates.append((a, a_t))
            updates.append((param, p_t))

        return updates


class MomentumSGD:
    def __init__(self, params, momentum=0.9, nesterov=False):
        params = [p() for p in params if p.status != 1]
        zero = np.array(0.0).astype(floatX)
        self._moments = [aesara.shared(p.get_value() * zero) for p in params]
        self.momentum = momentum
        self.nesterov = nesterov

    def update(self, cost, params, lr=0.001):
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        moments = self._moments
        for m, param, grad in zip(moments, params, grads):
            velocity = self.momentum * m - lr * grad

            if self.nesterov:
                p_t = param + self.momentum * velocity - lr * grad
            else:
                p_t = param + velocity

            updates.append((m, velocity))
            updates.append((param, p_t))

        return updates


class SGD:
    def __init__(self, params=None):
        params = [p() for p in params if p.status != 1]
        pass

    def update(self, cost, params, lr=0.001):
        params = [p() for p in params if p.status != 1]
        grads = aet.grad(cost, params, disconnected_inputs="ignore")

        updates = []
        for param, grad in zip(params, grads):
            p_t = param - lr * grad
            updates.append((param, p_t))

        return updates
