# expressions.py
"""PyCMTensor expressions module"""
import aesara
import aesara.tensor as aet
import numpy as np
from aesara import pprint
from aesara.tensor.sharedvar import TensorSharedVariable
from aesara.tensor.var import TensorVariable

FLOATX = aesara.config.floatX


class ExpressionParser:
    """Base class for the Expression Parser object"""

    def __init__(self):
        pass

    def parse(self, expression):
        """Returns a list of str words found in expression

        Args:
            expression: The symbolic Tensor object to parse
        """
        if isinstance(expression, str):
            stdout = expression
        else:
            stdout = str(pprint(expression))
        for s in [
            "(",
            ")",
            ",",
            "[",
            "]",
            "{",
            "}",
            "=",
            "-1",
            "Shape",
            "AdvancedSubtensor",
            "Reshape",
            "join",
            "sum",
            "dtype",
            "ARange",
            ":",
            "int64",
            "axis",
            "Softmax",
            "None",
            "log",
        ]:
            stdout = str.replace(stdout, s, " ")
        symbols = [s for s in str.split(stdout, " ") if len(s) > 1]
        symbols = list(set(symbols))
        return symbols


class Expressions:
    """Base class for expression objects"""

    def __init__(self):
        pass

    def __add__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return self() + other
        elif isinstance(other, Beta):
            return self() + other()
        else:
            raise NotImplementedError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __radd__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return other + self()
        elif isinstance(other, Beta):
            return other() + self()
        else:
            raise NotImplementedError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __sub__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return self() - other
        elif isinstance(other, Beta):
            return self() - other()
        else:
            raise NotImplementedError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __rsub__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return other - self()
        elif isinstance(other, Beta):
            return other() - self()
        else:
            raise NotImplementedError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __mul__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            if self().ndim > 1:
                return aet.dot(other, self().T)
            else:
                return self() * other
        elif isinstance(other, Beta):
            return self() * other()
        else:
            raise NotImplementedError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __rmul__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            if self().ndim > 1:
                return aet.dot(other, self().T)
            else:
                return self() * other
        elif isinstance(other, Beta):
            return self() * other()
        else:
            raise NotImplementedError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __div__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return self() / other
        elif isinstance(other, Beta):
            return self() / other()
        else:
            raise NotImplementedError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __rdiv__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return other / self()
        elif isinstance(other, Beta):
            return other() / self()
        else:
            raise NotImplementedError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __neg__(self):
        if isinstance(self, (TensorVariable, TensorSharedVariable)):
            return -self
        elif isinstance(self, Beta):
            return -self()
        else:
            raise NotImplementedError(
                f"{self} must be a TensorVariable or TensorShared Variable object"
            )

    def __pow__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return aet.pow(self(), other)
        elif isinstance(other, Beta):
            return aet.pow(self(), other())
        else:
            raise NotImplementedError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __rpow__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return aet.pow(other, self())
        elif isinstance(other, Beta):
            return aet.pow(other(), self())
        else:
            raise NotImplementedError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __lt__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return aet.lt(self(), other)
        elif isinstance(other, Beta):
            return aet.lt(self(), other())
        else:
            raise NotImplementedError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __le__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return aet.le(self(), other)
        elif isinstance(other, Beta):
            return aet.le(self(), other())
        else:
            raise NotImplementedError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __gt__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return aet.gt(self(), other)
        elif isinstance(other, Beta):
            return aet.gt(self(), other())
        else:
            raise NotImplementedError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __ge__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return aet.ge(self(), other)
        elif isinstance(other, Beta):
            return aet.ge(self(), other())
        else:
            raise NotImplementedError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __eq__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return aet.eq(self(), other)
        elif isinstance(other, Beta):
            return aet.eq(self(), other())
        else:
            raise NotImplementedError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __ne__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return aet.neq(self(), other)
        elif isinstance(other, Beta):
            return aet.neq(self(), other())
        else:
            raise NotImplementedError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )


class Beta(Expressions):
    """Class object for Beta parameters

    Attributes:
        name (str): Name of the Beta
        init_value (float): The inital value of the Beta when created
        status (int): The status of the Beta. Not modifiable
        shared_var (:class:`aet.TensorSharedVariable`): The shared variable object
            of the Beta parameter to be updated on. Can be accessed with a call function
    """

    def __init__(self, name, value=0.0, lowerbound=None, upperbound=None, status=0):
        """Constructor for Beta class object

        Args:
            name (str): Name of the Beta class object
            value (float): Initial starting value. Defaults to ``0``
            lowerbound (float): lowerbound value. Defaults to ``None``
            upperbound (float): upperbound value. Defaults to ``None``
            status (int): Whether to estimate (0) this Beta expression or not (1).
        """
        self._name = name
        self._init_value = value
        self._status = status
        self.lb = lowerbound
        self.ub = upperbound
        self.reset_value()

    @property
    def name(self):
        return self._name

    @property
    def init_value(self):
        return self._init_value

    @init_value.setter
    def init_value(self, value):
        self._init_value = value
        self.reset_value()

    @property
    def status(self):
        return self._status

    def __call__(self):
        """Returns the shared value of the Beta object"""
        return self.shared_var

    def __repr__(self):
        return f"{self.name}({self.shared_var.get_value()}, {type(self.shared_var)})"

    def get_value(self):
        """Returns the numpy representation of the Beta value"""
        return self.shared_var.get_value()

    def reset_value(self):
        """Resets the value of the shared variable to the initial value"""
        value = np.asarray(self.init_value, dtype=FLOATX)
        self.shared_var = aesara.shared(value, name=self.name, borrow=False)


class Weights(Expressions):
    def __init__(self, name, size, status, random_init=True):
        assert isinstance(size, (list, tuple))
        rng = np.random.default_rng()

        self.name = name
        self.size = size
        self.status = status
        self.random_init = random_init
        if len(size) == 1:
            value = np.zeros(size, dtype=FLOATX)
        else:
            if random_init:
                value = rng.uniform(
                    low=-np.sqrt(6.0 / sum(size)),
                    high=np.sqrt(6.0 / sum(size)),
                    size=size,
                )
            else:
                value = np.zeros(size, dtype=FLOATX)

        self.init_value = value
        self.shared_var = aesara.shared(value=value, name=name, borrow=True)
        self.shape = self.shared_var.shape

    def reset_value(self):
        """Resets the value of the shared variable to the initial value"""
        value = np.asarray(self.init_value, dtype=FLOATX)
        self.shared_var = aesara.shared(value, name=self.name, borrow=True)

    def __call__(self):
        return self.shared_var

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return f"{self.name}(shape{self.shareshared_vardVar.shape.eval()}, TensorSharedVariable)"

    def get_value(self):
        """Returns the numpy representation of the Weights"""
        return self.shared_var.get_value()
