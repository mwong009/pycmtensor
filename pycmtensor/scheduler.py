# scheduler.py
"""PyCMTensor scheduler module

This module contains the implementation of the learning rate scheduler. By default, a constant learning rate is used. 
"""

import numpy as np

__all__ = [
    "Scheduler",
    "ConstantLR",
    "StepLR",
    "PolynomialLR",
    "CyclicLR",
    "TriangularCLR",
    "ExpRangeCLR",
]


class Scheduler:
    def __init__(self, lr):
        """Base class for learning rate scheduler

        Args:
            lr (float): the base learning rate
        Attributes:
            history (list): (iteration #, lr) tuples
        """
        self.name = "Scheduler"
        self._base_lr = lr
        self._history = []

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        msg = f"{self.name}("
        attrs = [d for d in dir(self) if not d.startswith("_")]
        for a in attrs:
            if a == "name":
                continue
            if isinstance(getattr(self, a), (int, str, float)):
                msg += f"{a}={getattr(self, a)}, "

        return msg[:-2] + ")"

    def __call__(self, i):
        self.record(i, self.lr)
        return self.lr

    @property
    def lr(self):
        return self._base_lr

    @property
    def history(self):
        return self._history

    def record(self, iteration, lr):
        """Saves the history of the learning rate and returns the current learning rate

        Args:
            iteration (int): the interation number
            lr (float): the learning rate

        Returns:
            (float): the current learning rate
        """
        self.history.append((iteration, lr))
        return lr


class ConstantLR(Scheduler):
    def __init__(self, lr=0.01, **kwargs):
        """Base class for constant learning rate scheduler

        Args:
            lr (float): initial learning rate
            **kwargs (dict): overloaded keyword arguments
        """
        super().__init__(lr)
        self.name = "ConstantLR"


class StepLR(Scheduler):
    def __init__(self, lr=0.01, factor=0.95, drop_every=10, **kwargs):
        """Base class for step learning rate scheduler

        Args:
            lr (float): initial learning rate
            factor (float): percentage reduction to the learning rate
            drop_every (int): step down the learning rate after every n steps
            **kwargs (dict): overloaded keyword arguments
        """
        super().__init__(lr)
        self.name = "StepLR"
        self._factor = factor
        self._drop_every = drop_every
        self._min_lr = 0.01 * lr

        if factor >= 1.0:
            raise ValueError(f"factor is greater than 1.")

    @property
    def factor(self):
        return self._factor

    @property
    def drop_every(self):
        return self._drop_every

    def __call__(self, iteration):
        decay = self.factor ** np.floor(iteration / self._drop_every)
        lr = max(float(self.lr * decay), self._min_lr)
        return self.record(iteration, lr)


class PolynomialLR(Scheduler):
    def __init__(self, max_epochs, lr=0.01, power=1.0, **kwargs):
        """Base class for polynomial decay learning rate scheduler

        Args:
            lr (float): initial learning rate value
            max_epochs (int): the max number of training epochs
            power (float): the exponential factor to decay
            **kwargs (dict): overloaded keyword arguments
        """
        super().__init__(lr)
        self.name = "PolynomialLR"
        self._max_epochs = max_epochs
        self._min_lr = 0.01 * lr
        self._power = power

        if self.power < 0:
            raise ValueError(f"power is less than 0.")

    @property
    def power(self):
        return self._power

    @property
    def max_epochs(self):
        return self._max_epochs

    def __call__(self, iteration):
        decay = (1 - (iteration / float(self.max_epochs))) ** self.power
        lr = max(float(self.lr * decay), self._min_lr)
        return self.record(iteration, lr)


class CyclicLR(Scheduler):
    def __init__(self, lr=0.01, max_lr=0.1, cycle_steps=16, scale_fn=None, **kwargs):
        super().__init__(lr)
        self.name = "CyclicLR"
        self._max_lr = max_lr
        self._min_lr = 0.01 * lr
        self._cycle_steps = cycle_steps
        self._scale_fn = scale_fn

        if self.max_lr < self.lr:
            raise ValueError(f"max_lr is less than lr")

    @property
    def max_lr(self):
        return self._max_lr

    @property
    def cycle_steps(self):
        return self._cycle_steps

    def __call__(self, step):
        cycle = np.floor(1 + step / self.cycle_steps)
        x = np.abs(step / (self.cycle_steps / 2) - 2 * cycle + 1)
        height = (self.max_lr - self.lr) * self.scale_fn(cycle)
        lr = max(self.lr + height * np.maximum(0, 1 - x), self._min_lr)
        return self.record(step, lr)

    def scale_fn(self, k):
        return 1.0


class TriangularCLR(CyclicLR):
    def __init__(self, lr=0.01, max_lr=0.1, cycle_steps=16, **kwargs):
        """Base class for Triangular Cyclic LR scheduler. The scaling of the Triangular Cyclic function is:

        $$
        scale = \\frac{1}{2^{step-1}}
        $$

        Args:
            lr (float): initial learning rate value
            max_lr (float): peak learning rate value
            cycle_steps (int): the number of steps to complete a cycle
            **kwargs (dict): overloaded keyword arguments
        """
        super().__init__(lr, max_lr, cycle_steps, scale_fn=self.scale_fn)
        self.name = "TriangularCLR"

    def scale_fn(self, k):
        return float(1.0 / (2.0 ** (k - 1.0)))


class ExpRangeCLR(CyclicLR):
    def __init__(self, lr=0.01, max_lr=0.1, cycle_steps=16, gamma=0.5, **kwargs):
        """Base class for exponential range Cyclic LR scheduler. The scaling is:

        $$
        scale = \\gamma^{step}
        $$

        Args:
            lr (float): initial learning rate value
            max_lr (float): peak learning rate value
            cycle_steps (int): the number of steps to complete a cycle
            gamma (float): exponential parameter. Default=0.5
            **kwargs (dict): overloaded keyword arguments
        """
        super().__init__(lr, max_lr, cycle_steps, scale_fn=self.scale_fn)
        self.name = "ExpRangeCLR"
        self._gamma = gamma

        if self.gamma > 1:
            raise ValueError(f"gamma is greater than 1.")

    @property
    def gamma(self):
        return self._gamma

    def scale_fn(self, k):
        return self.gamma**k
