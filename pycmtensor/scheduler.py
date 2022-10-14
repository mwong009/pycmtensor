# scheduler.py
"""PyCMTensor scheduler module"""
from collections import OrderedDict

import numpy as np

__all__ = [
    "Scheduler",
    "ConstantLR",
    "StepLR",
    "PolynomialLR",
    "CyclicLR",
    "Triangular2CLR",
    "ExpRangeCLR",
]


class Scheduler:
    """Base class for Scheduler object"""

    def __init__(self):
        """Constructor for Scheduler class object"""
        self.name = "Scheduler"

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


class ConstantLR(Scheduler):
    """Base class for constant learning rate scheduler"""

    def __init__(self, lr=0.01):
        """Constructor for ConstantLR class object

        Args:
            lr (float, optional): learning rate value
        """
        self.name = "ConstantLR"
        self._base_lr = lr
        self._history = OrderedDict()

    @property
    def lr(self):
        """Returns the learning rate value"""
        return self._base_lr

    @property
    def history(self):
        """Returns the histroy of the learning rate"""
        return self._history

    def __call__(self, step):
        """Computes the learning rate for this step"""
        self._record(step, self.lr)
        return self.lr

    def _record(self, step, lr):
        """Saves the history of the learning rate and returns the current rate"""
        self._history[step] = lr
        return lr


class StepLR(ConstantLR):
    """Base class for step learning rate scheduler"""

    def __init__(self, lr=0.01, factor=0.25, drop_every=10):
        """Constructor for StepLR class object

        Args:
            lr (float): initial learning rate value
            factor (float, optional): percentage reduction to the learning rate
            drop_every (int, optional): step down the learning rate after every n steps
        """
        super().__init__(lr)
        self.name = "StepLR"
        self._factor = factor
        self._drop_every = drop_every

        if factor >= 1.0:
            raise ValueError(f"factor is greater than 1.")

        if not isinstance(drop_every, int):
            raise ValueError(f"drop_every is not an integer.")

    @property
    def factor(self):
        """Returns the step factor value"""
        return self._factor

    @property
    def drop_every(self):
        """Returns the step distance value"""
        return self._drop_every

    def __call__(self, step):
        """Computes the learning rate for this step"""
        decay = self.factor ** np.floor(step / self._drop_every)
        lr = float(self.lr * decay)
        return self._record(step, lr)


class PolynomialLR(ConstantLR):
    """Base class for polynomial decay learning rate scheduler"""

    def __init__(self, max_steps, lr=0.01, power=1.0):
        """Constructor for PolynomialLR class object

        Args:
            lr (float): initial learning rate value
            max_steps (int): the max number of training steps to take
            power (float, optional): the exponential factor to decay
        """
        super().__init__(lr)
        self.name = "PolynomialLR"
        self._max_steps = max_steps
        self._power = power

        if self.power < 0:
            raise ValueError(f"power is less than 0.")

    @property
    def power(self):
        """Returns the exponent of the polynomial"""
        return self._power

    @property
    def max_steps(self):
        """Returns the max steps value"""
        return self._max_steps

    def __call__(self, step):
        """Computes the learning rate for this step"""
        decay = (1 - (step / float(self.max_steps))) ** self.power
        lr = float(self.lr * decay)
        return self._record(step, lr)


class CyclicLR(ConstantLR):
    """Base class for cyclical learning rate scheduler"""

    def __init__(self, lr=0.01, max_lr=0.1, cycle_steps=16, scale_fn=None):
        """Constructor for ConstantLR class object

        Args:
            lr (float, optional): the base learning rate value
            max_lr (float, optional): the maximum learning rate value
            cycle_steps (int, optional): the number of steps to complete a cycle
            scale_fn (func, optional): custom scaling policy defined by a single arg
        """
        super().__init__(lr)
        self.name = "CyclicLR"
        self._max_lr = max_lr
        self._cycle_steps = cycle_steps
        self._scale_fn = scale_fn

        if self.max_lr < self.lr:
            raise ValueError(f"max_lr is less than lr")

    @property
    def max_lr(self):
        """Returns the maximum learning rate value"""
        return self._max_lr

    @property
    def cycle_steps(self):
        """Returns the cycle steps value"""
        return self._cycle_steps

    def __call__(self, step):
        """Computes the learning rate for this step"""
        cycle = np.floor(1 + step / self.cycle_steps)
        x = np.abs(step / (self.cycle_steps / 2) - 2 * cycle + 1)
        height = (self.max_lr - self.lr) * self.scale_fn(cycle)
        lr = self.lr + height * np.maximum(0, 1 - x)
        return self._record(step, lr)

    def scale_fn(self, k):
        """Custom scaling policy"""
        if self._scale_fn is None:
            return 1.0
        else:
            return self._scale_fn(k)


class Triangular2CLR(CyclicLR):
    """Class object for the Triangular2 Cyclic LR scheduler"""

    def __init__(self, lr=0.01, max_lr=0.1, cycle_steps=16):
        """Constructor for Triangular2CLR class object

        Args:
            lr (float, optional): the base learning rate value
            max_lr (float, optional): the maximum learning rate value
            cycle_steps (int, optional): the number of steps to complete a cycle
        """
        super().__init__(lr, max_lr, cycle_steps)
        self.name = "Triangular2CLR"

    def scale_fn(self, k):
        """Calculates the cycle amplitude scale"""
        return float(1.0 / (2.0 ** (k - 1.0)))


class ExpRangeCLR(CyclicLR):
    """Class object for the exponential range Cyclic LR scheduler"""

    def __init__(self, lr=0.01, max_lr=0.1, cycle_steps=16, gamma=0.5):
        """Constructor for Triangular2CLR class object

        Args:
            lr (float, optional): the base learning rate value
            max_lr (float, optional): the maximum learning rate value
            cycle_steps (int, optional): the number of steps to complete a cycle
        """
        super().__init__(lr, max_lr, cycle_steps)
        self.name = "ExpRangeCLR"
        self._gamma = gamma

        if self.gamma > 1:
            raise ValueError(f"gamma is greater than 1.")

    @property
    def gamma(self):
        """Returns the gamma value"""
        return self._gamma

    def scale_fn(self, k):
        """Calculates the cycle amplitude scale"""
        return self.gamma**k
