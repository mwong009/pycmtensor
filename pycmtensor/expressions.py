import aesara
import aesara.tensor as aet
import biogeme.expressions as bioexp
import numpy as np
from aesara.tensor.var import TensorVariable

floatX = aesara.config.floatX


class Expressions:
    def __init__(self):
        pass

    def __add__(self, other):
        if isinstance(other, (TensorVariable, Beta)):
            return self.sharedVar + other
        return super().__add__(other)

    def __radd__(self, other):
        if isinstance(other, (TensorVariable, Beta)):
            return self.sharedVar + other
        return super().__radd__(other)

    def __sub__(self, other):
        if isinstance(other, (TensorVariable, Beta)):
            return self.sharedVar - other
        return super().__sub__(other)

    def __rsub__(self, other):
        if isinstance(other, (TensorVariable, Beta)):
            return other - self.sharedVar
        return super().__rsub__(other)

    def __mul__(self, other):
        if isinstance(other, (TensorVariable, Beta)):
            if self.sharedVar.ndim > 1:
                return aet.dot(other, self.sharedVar.T)
            else:
                return self.sharedVar * other
        return super().__mul__(other)

    def __rmul__(self, other):
        if isinstance(other, (TensorVariable, Beta)):
            if self.sharedVar.ndim > 1:
                return aet.dot(other, self.sharedVar.T)
            else:
                return other * self.sharedVar
        return super().__rmul__(other)

    def __div__(self, other):
        if isinstance(other, (TensorVariable, Beta)):
            return self.sharedVar / other
        return super().__div__(other)

    def __rdiv__(self, other):
        if isinstance(other, (TensorVariable, Beta)):
            return other / self.sharedVar
        return super().__rdiv__(other)

    def __truediv__(self, other):
        if isinstance(other, (TensorVariable, Beta)):
            return self.sharedVar / other
        return super().__truediv__(other)

    def __rtruediv__(self, other):
        if isinstance(other, (TensorVariable, Beta)):
            return other / self.sharedVar
        return super().__rtruediv__(other)

    def __neg__(self):
        if isinstance(self, (TensorVariable, Beta)):
            return -self
        return super().__neg__()

    def __pow__(self, other):
        if isinstance(other, (TensorVariable, Beta)):
            return self.sharedVar**other
        return super().__pow__(other)

    def __rpow__(self, other):
        if isinstance(other, (TensorVariable, Beta)):
            return other**self.sharedVar
        return super().__pow__(other)


class Beta(Expressions, bioexp.Beta):
    def __init__(self, name, value, lb, ub, status):
        bioexp.Beta.__init__(self, name, value, lb, ub, status)
        self.sharedVar = aesara.shared(value=np.array(value, dtype=floatX), name=name)
        self.sharedVar.__dict__.update({"status": status, "lb": lb, "ub": ub})

    def __call__(self):
        return self.sharedVar

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return f"{self.name}({self.sharedVar.get_value()}, TensorSharedVariable)"


class Weights(Expressions):
    def __init__(self, name, size, status, random_init=True):
        assert isinstance(size, (list, tuple))
        rng = np.random.default_rng()

        self.name = name
        self.size = size
        self.status = status
        self.random_init = random_init
        if len(size) == 1:
            value = np.zeros(size, dtype=floatX)
        else:
            if random_init:
                value = rng.uniform(
                    low=-np.sqrt(6.0 / sum(size)),
                    high=np.sqrt(6.0 / sum(size)),
                    size=size,
                )
            else:
                value = np.zeros(size, dtype=floatX)

        self.sharedVar = aesara.shared(value=value, name=name)
        self.shape = self.sharedVar.shape
        self.sharedVar.__dict__.update({"status": self.status})

    def __call__(self):
        return self.sharedVar

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return f"{self.name}(shape{self.sharedVar.shape.eval()}, TensorSharedVariable)"

    def get_value(self):
        return self.sharedVar.get_value()
