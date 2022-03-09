import numpy as np


class ConstantLR:
    def __init__(self, base_lr=0.01, **kwargs):
        self.base_lr = base_lr

    def get_lr(self):
        return self.base_lr


class CyclicLR(ConstantLR):
    def __init__(
        self, base_lr=0.001, max_lr=0.01, step_size=8, mode="triangular2", gamma=1.0
    ):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = float(step_size)

        if mode not in ["triangular", "triangular2", "exp_range"]:
            raise ValueError(f"{mode} is an invalid option for mode")

        self.mode = mode
        self.gamma = gamma

        if self.mode == "triangular":
            self.scale_fn = self._triangular_scale_fn
        elif self.mode == "triangular2":
            self.scale_fn = self._triangular2_scale_fn
        elif self.mode == "exp_range":
            self.scale_fn = self._exp_range_scale_fn

    def _triangular_scale_fn(self, x):
        return 1.0

    def _triangular2_scale_fn(self, x):
        return 1.0 / (2.0 ** (x - 1.0))

    def _exp_range_scale_fn(self, x):
        return self.gamma**x

    def get_lr(self, epoch_counter):
        """Calculates the learning rate given the epoch

        Args:
            epoch_counter (int): The current epoch count

        Returns:
            float: the learning rate to apply
        """
        cycle = np.floor(1 + epoch_counter / (2 * self.step_size))
        x = np.abs(epoch_counter / self.step_size - 2 * cycle + 1.0)
        base_height = (self.max_lr - self.base_lr) * self.scale_fn(cycle)
        lr = self.base_lr + base_height * np.maximum(0, 1 - x)
        return lr
