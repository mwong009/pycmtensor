# scheduler.py
"""PyCMTensor scheduler module

The code snippet defines a base class called `Scheduler` for learning rate schedulers. It also includes three subclasses: `ConstantLR`, `StepLR`, and `PolynomialLR`, which implement specific learning rate scheduling strategies.

Example Usage:
- Creating a `Scheduler` object:
    scheduler = Scheduler(lr=0.01)
- Getting the learning rate for a specific epoch:
    lr = scheduler(epoch=5)
- Creating a `ConstantLR` object:
    constant_lr = ConstantLR(lr=0.01)
- Getting the learning rate for a specific epoch:
    lr = constant_lr(epoch=10)
- Creating a `StepLR` object:
    step_lr = StepLR(lr=0.01, factor=0.95, drop_every=5)
- Getting the learning rate for a specific epoch:
    lr = step_lr(epoch=15)
- Creating a `PolynomialLR` object:
    poly_lr = PolynomialLR(lr=0.01, max_epochs=20, power=0.5)
- Getting the learning rate for a specific epoch:
    lr = poly_lr(epoch=8)
"""

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
    def __init__(self, lr=0.01, epochs=2000):
        """Initializes the Scheduler object with a base learning rate.

        Args:
            lr (float): The base learning rate.

        Attributes:
            name (str): Name of the scheduler.
            _base_lr (float): Base learning rate.
            _history (list): List to store the learning rate history.
        """
        self.name = "Scheduler"
        self._base_lr = lr
        self._learning_rate = lr
        self._history = []
        self.max_epochs = epochs

    def __str__(self):
        """Returns a string representation of the Scheduler object.

        Returns:
            (str): String representation of the Scheduler object.
        """
        return f"{self.name}"

    def __repr__(self):
        """Returns a string representation of the Scheduler object with its attributes.

        Returns:
            (str): String representation of the Scheduler object with its attributes.
        """
        msg = f"{self.name}("
        attrs = [d for d in dir(self) if not d.startswith("_")]
        for a in attrs:
            if a == "name":
                continue
            if isinstance(getattr(self, a), (int, str, float)):
                msg += f"{a}={getattr(self, a)}, "

        return msg[:-2] + ")"

    def __call__(self, **kwargs):
        """Returns the current learning rate.

        Returns:
            (float): The current learning rate.
        """
        return self.learning_rate

    @property
    def learning_rate(self):
        """Property that returns the current learning rate.

        Returns:
            (float): The current learning rate.
        """
        return self._learning_rate

    @property
    def lr(self):
        """Property that returns the base learning rate.

        Returns:
            (float): The base learning rate.
        """
        return self._base_lr

    @property
    def history(self):
        """Property that returns the learning rate history.

        Returns:
            (list): The learning rate history.
        """
        return self._history

    def record(self, lr):
        """Saves the history of the learning rate and returns the current learning rate.

        Args:
            lr (float): The learning rate.

        Returns:
            (float): The current learning rate.
        """
        self.history.append(lr)
        return lr


class ConstantLR(Scheduler):
    def __init__(self, max_epochs=2000, lr=0.01):
        """Subclass of Scheduler for constant learning rate scheduler.

        Args:
            lr (float): initial learning rate
        """
        super().__init__(lr, epochs=max_epochs)
        self.name = "ConstantLR"


class StepLR(Scheduler):
    def __init__(self, max_epochs=2000, lr=0.01, factor=0.95, drop=20):
        """Base class for step learning rate scheduler

        Args:
            lr (float): initial learning rate
            factor (float): percentage reduction to the learning rate
            drop (int): step down the learning rate after every n epochs
        """
        super().__init__(lr, epochs=max_epochs)
        self.name = "StepLR"
        self._factor = factor
        self._drop = drop
        self._min_lr = 0.01 * lr  # minimum learning rate = 1% of initial learning rate

        if factor >= 1.0:
            raise ValueError(f"factor is greater than 1.")

    @property
    def factor(self):
        return self._factor

    @property
    def drop(self):
        return self._drop

    def __call__(self, epoch):
        decay = self.factor ** np.floor(epoch / self._drop)
        self._learning_rate = max(float(self.lr * decay), self._min_lr)

        return self.learning_rate


class PolynomialLR(Scheduler):
    def __init__(self, max_epochs=2000, lr=0.01, power=1.0):
        """Subclass of Scheduler for polynomial decay learning rate scheduler.

        Args:
            max_epochs (int): the max number of training epochs
            lr (float): initial learning rate value
            power (float): the exponential factor to decay
        """
        super().__init__(lr, epochs=max_epochs)
        self.name = "PolynomialLR"
        self._min_lr = 0.01 * lr
        self._power = power

        if self.power < 0:
            raise ValueError(f"power is less than 0.")

    @property
    def power(self):
        return self._power

    def __call__(self, epoch):
        decay = (1 - (epoch / float(self.max_epochs))) ** self.power
        self._learning_rate = max(float(self.lr * decay), self._min_lr)

        return self.learning_rate


class CyclicLR(Scheduler):
    def __init__(self, max_epochs=2000, lr=0.01, cycle_steps=16):
        """Subclass of Scheduler for cyclic learning rate scheduler.

        Args:
            lr (float, optional): initial learning rate value.
            cycle_steps (int): The number of steps to complete a cycle.

        Raises:
            ValueError: _description_
        """
        super().__init__(lr, epochs=max_epochs)
        self.name = "CyclicLR"
        self._max_lr = lr * 10
        self._cycle_steps = cycle_steps

    @property
    def max_lr(self):
        return self._max_lr

    @property
    def cycle_steps(self):
        return self._cycle_steps

    def __call__(self, epoch):
        cycle = np.floor(1 + epoch / (2 * self.cycle_steps))
        x = np.abs(epoch / self.cycle_steps - 2 * cycle + 1)
        height = (self.max_lr - self.lr) * self.scale_fn(cycle)
        lr = self.lr + height * np.maximum(0, 1 - x)
        return self.record(lr)

    def scale_fn(self, k):
        return 1.0


class Triangular2CLR(CyclicLR):
    def __init__(self, max_epochs=2000, lr=0.01, cycle_steps=16):
        """Subclass of CyclicLR for Triangular Cyclic LR scheduler.

        The scaling of the Triangular Cyclic function is:

        $$
        scale = \\frac{1}{2^{step-1}}
        $$

        Args:
            lr (float, optional): initial learning rate value.
            max_lr (float): Peak learning rate value.
            cycle_steps (int): The number of steps to complete a cycle.
        """
        super().__init__(max_epochs, lr, cycle_steps)
        self.name = "Triangular2CLR"

    def scale_fn(self, k):
        return float(1.0 / (2.0 ** (k - 1.0)))


class ExpRangeCLR(CyclicLR):
    def __init__(self, max_epochs=2000, lr=0.01, cycle_steps=16, gamma=0.5):
        """Subclass of CyclicLR for exponential range Cyclic LR scheduler.

        The scaling is:

        $$
        scale = \\gamma^{step}
        $$

        Args:
            lr (float, optional): initial learning rate value.
            max_lr (float): Peak learning rate value.
            cycle_steps (int): The number of steps to complete a cycle.
            gamma (float): Exponential parameter.
        """
        super().__init__(max_epochs, lr, cycle_steps)
        self.name = "ExpRangeCLR"
        self._gamma = gamma

        if self.gamma > 1:
            raise ValueError(f"gamma is greater than 1.")

    @property
    def gamma(self):
        return self._gamma

    def scale_fn(self, k):
        return self.gamma**k
