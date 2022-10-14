# expressions.py
"""PyCMTensor expressions module"""
import aesara
import aesara.tensor as aet
import numpy as np
from aesara import pprint
from aesara.tensor.random.utils import RandomStream
from aesara.tensor.sharedvar import TensorSharedVariable
from aesara.tensor.var import TensorVariable

from .logger import log

FLOATX = aesara.config.floatX


class ExpressionParser:
    """Base class for the ExpressionParser object"""

    def __init__(self):
        pass

    def parse(self, expression):
        """Returns a list of str words found in expression

        Args:
            expression (TensorVariable): the symbolic Tensor object to parse
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


class ModelParam:
    def __init__(self, name: str, rng=None):
        """Constructor for model param object"""
        self._name = name

        if rng is None:
            self.rng = np.random.default_rng()
        else:
            assert isinstance(rng, np.random.Generator)
            self.rng = rng

    @property
    def name(self):
        """Returns the name of the object"""
        return self._name

    def __call__(self):
        """Returns the shared value"""
        return self.shared_var

    def __repr__(self):
        return f"{self.name}({self.shared_var.get_value()}, {type(self.shared_var)})"

    def get_value(self):
        """Returns the numpy representation of the Beta value"""
        return self.shared_var.get_value()

    def reset_value(self):
        """Resets the value of the shared variable to the initial value"""
        self.shared_var = aesara.shared(self.init_value, name=self.name, borrow=False)


class Beta(Expressions, ModelParam):
    def __init__(self, name, value=0.0, lb=None, ub=None, status=0):
        """Class object for Beta parameters

        Args:
            name (str): name of the Beta class object
            value (float): initial starting value. Defaults to ``0``
            lb (float): lowerbound value. Defaults to ``None``
            ub (float): upperbound value. Defaults to ``None``
            status (int): whether to estimate (0) this Beta expression or not (1).
        """
        ModelParam.__init__(self, name)
        self._status = status
        self.lb = lb
        self.ub = ub
        self._init_value = np.asarray(value, dtype=FLOATX)
        self.reset_value()

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

    @property
    def beta(self):
        return self.shared_var


class Sigma(Beta):
    def __init__(self, name, value=1.0, ub=None, status=0, dist="NORMAL"):
        super().__init__(name, value, 0, ub, status)
        self._dist = dist
        self.srng = RandomStream(seed=42069)

    @property
    def dist(self):
        return self._dist

    @property
    def sigma(self):
        return self.shared_var


class Weights(Expressions, ModelParam):
    def __init__(self, name, size, init_type=None, init_value=None, rng=None):
        """Class object for Neural Network weights

        Args:
            name (str): name of the weight
            size (tuple, list): array size of the weight, ndim=2
            init_type (str): initialization type, see notes
            init_value (numpy.ndarray, optional): initial value of the weights
            rng (numpy.random.Generator, optional): random generator

        Note:
            Initialization types are one of the following:

            * "he": initialization method for neural networks that takes into account
              the non-linearity of activation functions, e.g. ReLU or Softplus [#]_

            * "glorot": initialization method that maintains the variance for
              symmetric activation functions, e.g. sigm, tanh [#]_

            .. [#] He, K., Zhang, X., Ren, S. and Sun, J., 2015. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision (pp. 1026-1034).
            .. [#] Glorot, X. and Bengio, Y., 2010, March. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics (pp. 249-256). JMLR Workshop and Conference Proceedings.
        """
        ModelParam.__init__(self, name, rng)
        if not ((len(size) == 2) and isinstance(size, (list, tuple))):
            raise ValueError(
                f"Invalid size argument, len(size)={len(size)}, type(size)={type(size)}"
            )

        n_in, n_out = size

        if init_value is None:
            if init_type is None:
                log(10, f"using default initialization for {name}")
                init_value = self.rng.uniform(-1.0, 1.0, size=size)
            elif init_type == "he":
                init_value = self.rng.normal(size=size) * np.sqrt(2 / n_in)
            elif init_type == "glorot":
                init_value = self.rng.uniform(-1.0, 1.0, size) * np.sqrt(
                    6 / (n_in + n_out)
                )
            else:
                log(
                    10,
                    f'init_type {name} not implemented yet. Options: "he" or "glorot"',
                )

        if not init_value.shape == size:
            raise ValueError(f"init_value argument is not a valid array of size {size}")

        self._init_value = init_value
        self._shape = size
        self.reset_value()

    @property
    def init_value(self):
        return self._init_value

    @property
    def shape(self):
        return self._shape

    @property
    def W(self):
        return self.shared_var
