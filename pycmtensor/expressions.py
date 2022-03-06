import aesara
import aesara.tensor as aet
import biogeme.expressions as bioexp
import numpy as np
from aesara.tensor.sharedvar import TensorSharedVariable
from aesara.tensor.var import TensorVariable

floatX = aesara.config.floatX


class Expressions:
    def __init__(self):
        pass

    def __add__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return self() + other
        elif isinstance(other, Beta):
            return self() + other()
        else:
            return super().__add__(other)

    def __radd__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return other + self()
        elif isinstance(other, Beta):
            return other() + self()
        else:
            return super().__radd__(other)

    def __sub__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return self() - other
        elif isinstance(other, Beta):
            return self() - other()
        else:
            return super().__sub__(other)

    def __rsub__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return other - self()
        elif isinstance(other, Beta):
            return other() - self()
        else:
            return super().__rsub__(other)

    def __mul__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            if self().ndim > 1:
                return aet.dot(other, self().T)
            else:
                return self() * other
        elif isinstance(other, Beta):
            return self() * other()
        else:
            return super().__mul__(other)

    def __rmul__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            if self.sharedVar.ndim > 1:
                return aet.dot(other, self().T)
            else:
                return self() * other
        elif isinstance(other, Beta):
            return self() * other()
        else:
            return super().__rmul__(other)

    def __div__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return self() / other
        elif isinstance(other, Beta):
            return self() / other()
        else:
            return super().__div__(other)

    def __rdiv__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return other / self()
        elif isinstance(other, Beta):
            return other() / self()
        else:
            return super().__rdiv__(other)

    def __neg__(self):
        if isinstance(self, (TensorVariable, TensorSharedVariable)):
            return -self
        elif isinstance(self, Beta):
            return -self()
        return super().__neg__(self)

    def __pow__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return aet.pow(self(), other)
        elif isinstance(other, Beta):
            return aet.pow(self(), other())
        else:
            return super().__pow__(other)

    def __rpow__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):

            return aet.pow(other, self())
        elif isinstance(other, Beta):
            return aet.pow(other(), self())
        else:
            return super().__rpow__(other)

    def __lt__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return aet.lt(self(), other)
        elif isinstance(other, Beta):
            return aet.lt(self(), other())
        else:
            return super().__lt__(other)

    def __le__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return aet.le(self(), other)
        elif isinstance(other, Beta):
            return aet.le(self(), other())
        else:
            return super().__le__(other)

    def __gt__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return aet.gt(self(), other)
        elif isinstance(other, Beta):
            return aet.gt(self(), other())
        else:
            return super().__gt__(other)

    def __ge__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return aet.ge(self(), other)
        elif isinstance(other, Beta):
            return aet.ge(self(), other())
        else:
            return super().__ge__(other)

    def __eq__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return aet.eq(self(), other)
        elif isinstance(other, Beta):
            return aet.eq(self(), other())
        else:
            return super().__eq__(other)

    def __ne__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return aet.neq(self(), other)
        elif isinstance(other, Beta):
            return aet.neq(self(), other())
        else:
            return super().__ne__(other)


class Beta(Expressions, bioexp.Beta):
    def __init__(self, name, value, lb, ub, status):
        bioexp.Beta.__init__(self, name, value, lb, ub, status)
        self.sharedVar = aesara.shared(
            value=np.array(value, dtype=floatX), borrow=True, name=name
        )
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

        self.sharedVar = aesara.shared(value=value, name=name, borrow=True)
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
