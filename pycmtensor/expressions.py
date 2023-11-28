"""
The code snippet is a part of the PyCMTensor expressions module. It defines a base class for parsing and manipulating Aesara tensor expressions. The class provides methods for parsing a tensor expression to remove parentheses and tensor operators, and returns a clean list of keywords found in the expression. It also defines a base class for expression objects, which includes overloaded operators for tensor operations such as addition, subtraction, multiplication, division, and comparison.
"""
from typing import Union

import aesara
import aesara.tensor as aet
import numpy as np
from aesara import pprint
from aesara.tensor.random.utils import RandomStream
from aesara.tensor.sharedvar import TensorSharedVariable
from aesara.tensor.var import TensorVariable

from pycmtensor.logger import debug

__all__ = ["FLOATX", "Param", "Beta", "Sigma", "Bias", "Weight"]
FLOATX = aesara.config.floatX


class ExpressionParser(object):
    HARD_CODED_STRINGS = [
        "(",
        ")",
        ",",
        "[",
        "]",
        "{",
        "}",
        "=",
        "*",
        "-",
        "+",
        "/",
        ":",
        "AdvancedSubtensor",
        "Reshape",
        "ARange",
        "Assert",
        "Shape",
        "BroadcastTo",
        "Composite",
        "Could",
        "ScalarFromTensor",
        "Abs",
        "Softmax",
        "Switch",
        "dtype",
    ]

    def __init__(self, expression=None):
        """Base class for parsing and manipulating Aesara tensor expressions.

        Args:
            expression (TensorVariable, optional): The tensor expression to parse. Defaults to None.
        """
        if expression is not None:
            self.expression = str(pprint(expression))

    @staticmethod
    def parse(expression):
        """Parses a tensor expression to remove parentheses and tensor operators.

        Args:
            expression (TensorVariable): The symbolic tensor object to parse.

        Returns:
            list: The clean list of keywords found in the expression.
        """
        stdout = ExpressionParser._get_stdout(expression)
        stdout = ExpressionParser._remove_parentheses(stdout)
        stdout = ExpressionParser._remove_tensor_operators(stdout)
        symbols = ExpressionParser._remove_duplicates(stdout)
        return symbols

    @staticmethod
    def _get_stdout(expression):
        if isinstance(expression, str):
            return expression
        else:
            return str(pprint(expression))

    @staticmethod
    def _remove_parentheses(stdout):
        for s in ["(", ")", "[", "]", "{", "}"]:
            stdout = stdout.replace(s, " ")
        return stdout

    @staticmethod
    def _remove_tensor_operators(stdout):
        for s in ExpressionParser.HARD_CODED_STRINGS:
            stdout = stdout.replace(s, " ")
        return stdout

    @staticmethod
    def _remove_duplicates(stdout):
        symbols = [s for s in stdout.split(" ") if len(s) > 0]
        return list(set(symbols))


class TensorExpressions:
    """Base class for expression objects"""

    def __init__(self):
        pass

    def __add__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return self() + other
        elif isinstance(other, Param):
            return self() + other()
        else:
            raise TypeError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __radd__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return other + self()
        elif isinstance(other, Param):
            return other() + self()
        else:
            raise TypeError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __sub__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return self() - other
        elif isinstance(other, Param):
            return self() - other()
        else:
            raise TypeError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __rsub__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return other - self()
        elif isinstance(other, Param):
            return other() - self()
        else:
            raise TypeError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __mul__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            if self().ndim > 1:
                return aet.dot(other, self().T)
            else:
                return self() * other
        elif isinstance(other, (Param, RandomDraws)):
            return self() * other()
        else:
            raise TypeError(
                f"__mul__ {other} must be a TensorVariable or TensorShared Variable object"
            )

    def __rmul__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            if self().ndim > 1:
                return aet.dot(other, self().T)
            else:
                return self() * other
        elif isinstance(other, (Param, RandomDraws)):
            return self() * other()
        else:
            raise TypeError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __div__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return self() / other
        elif isinstance(other, Param):
            return self() / other()
        else:
            raise TypeError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __rdiv__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return other / self()
        elif isinstance(other, Param):
            return other() / self()
        else:
            raise TypeError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __neg__(self):
        return -self()

    def __pow__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return aet.pow(self(), other)
        elif isinstance(other, Param):
            return aet.pow(self(), other())
        else:
            raise TypeError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __rpow__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return aet.pow(other, self())
        elif isinstance(other, Param):
            return aet.pow(other(), self())
        else:
            raise TypeError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __lt__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return aet.lt(self(), other)
        elif isinstance(other, Param):
            return aet.lt(self(), other())
        else:
            raise TypeError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __le__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return aet.le(self(), other)
        elif isinstance(other, Param):
            return aet.le(self(), other())
        else:
            raise TypeError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __gt__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return aet.gt(self(), other)
        elif isinstance(other, Param):
            return aet.gt(self(), other())
        else:
            raise TypeError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __ge__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return aet.ge(self(), other)
        elif isinstance(other, Param):
            return aet.ge(self(), other())
        else:
            raise TypeError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __eq__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return aet.eq(self(), other)
        elif isinstance(other, Param):
            return aet.eq(self(), other())
        else:
            raise TypeError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )

    def __ne__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            return aet.neq(self(), other)
        elif isinstance(other, Param):
            return aet.neq(self(), other())
        else:
            raise TypeError(
                f"{other} must be a TensorVariable or TensorShared Variable object"
            )


class Param(TensorExpressions):
    def __init__(self, name, value=0.0, lb=None, ub=None, status=0):
        """Constructor for model param object

        Args:
            name (str): name of parameter
            value (float): the default value of the parameter
            lb (float): value lower bound
            ub (float): value upper bound
            status (int): if 1, do not estimate this parameter

        Attributes:
            init_value (float): the inital value set at object creation
            shape (list): the shape of the Param

        Raises:
            ValueError: If `lb` is greater than `ub`

        !!! note
            `init_value` is an immutable property
        """
        self._name = name
        self._status = status
        if not isinstance(value, np.ndarray):
            value = np.asarray(value)
        self.shared_var = aesara.shared(value, name=name, borrow=True)
        self._init_value = value

        if lb is not None and ub is not None and lb > ub:
            raise ValueError("lb must be less than or equal to ub")
        self.ub = ub
        self.lb = lb

    @property
    def name(self):
        """Returns the name of the object"""
        return self._name

    @property
    def init_value(self):
        """Returns the initial value of the parameter"""
        return self._init_value

    @property
    def status(self):
        """Returns the status of the parameter"""
        return self._status

    @property
    def shape(self):
        """Returns the shape of the initial value of the parameter"""
        return self._init_value.shape

    @property
    def T(self):
        raise NotImplementedError

    @property
    def init_type(self):
        raise NotImplementedError

    def __call__(self):
        """Returns the shared value of the parameter"""
        return self.shared_var

    def __repr__(self):
        """Returns a string representation of the parameter"""
        return f"Param({self.name}, {self.shape})"

    def get_value(self):
        """Returns the numpy representation of the parameter value"""
        return self.shared_var.get_value()

    def set_value(self, value: Union[np.ndarray, float]):
        """Sets the value of the shared variable"""
        self.shared_var.set_value(value)

    def reset_value(self):
        """Resets the value of the shared variable to the initial value"""
        self.shared_var.set_value(self.init_value)


class Beta(Param):
    def __init__(self, name, value=0.0, lb=None, ub=None, status=0):
        """Constructor for Beta parameter

        Args:
            name (str): The name of the Beta parameter.
            value (float): The default value of the Beta parameter.
            lb (float): The lower bound of the Beta parameter.
            ub (float): The upper bound of the Beta parameter.
            status (int): The status of the Beta parameter.

        !!! example
            Specifying a Beta parameter:

            ```python
            b_time = Beta("b_time", value=0., lb=None, ub=0., status=0)
            ```

            To obtain the raw array value from the Beta parameter:
            ```python
            b_time = Beta("b_time", value=-1., lb=None, ub=0., status=0)
            b_time.get_value()
            ```

            output:
            ```console
            array(-1.)
            ```
        """
        Param.__init__(self, name, lb=lb, ub=ub, status=status)

        self._init_value = np.asarray(value, dtype=FLOATX)
        self.shared_var = aesara.shared(self.init_value, name=name, borrow=True)

    def __repr__(self):
        return f"Beta({self.name}, {self.get_value()})"


class RandomDraws(TensorExpressions):
    def __init__(self, name: str, draw_type: str, n_draws: int):
        """Constructor for the RandomDraws class.

        Args:
            name (str): The name of the RandomDraw object.
            draw_type (str): The distribution of the draw. Can be "normal", "lognormal", "gumbel", "exponential", "gamma", or "poisson".
            n_draws (int): The number of draws, which determines the size of the shared tensor.

        Raises:
            NotImplementedError: If an unsupported draw_type is provided.

        Returns:
            None. The method initializes the object and creates a shared variable.
        """
        self._name = name
        self.n_draws = n_draws
        srng = RandomStream()
        draw_type = draw_type.lower()
        if draw_type == "normal":
            rv_n = srng.normal(size=(n_draws, 1))
        elif draw_type == "lognormal":
            rv_n = srng.lognormal(size=(n_draws, 1))
        elif draw_type == "gumbel":
            rv_n = srng.gumbel(loc=0.0, scale=1.0, size=(n_draws, 1))
        elif draw_type == "exponential":
            rv_n = srng.exponential(scale=1.0, size=(n_draws, 1))
        elif draw_type == "gamma":
            rv_n = srng.gamma(shape=1.0, rate=1.0, size=(n_draws, 1))
        elif draw_type == "poisson":
            rv_n = srng.poisson(lam=1.0, size=(n_draws, 1))
        else:
            raise NotImplementedError
        self.shared_var = aesara.shared(rv_n.eval(), name=self.name)

    @property
    def name(self):
        """returns the name of the random draw tensor variable"""
        return self._name

    def __repr__(self):
        return f"RandomDraws({self.name}, size=({self.n_draws}, 1))"

    def __call__(self):
        return self.shared_var


class Bias(Param):
    def __init__(self, name, size, value=None):
        """Class object for neural net bias vector

        Args:
            name (str): The name of the parameter.
            size (Union[tuple,list]): The size of the array in 1 dimension.
            value (numpy.ndarray): The initial values of the parameter. If `None` is given, it defaults to `0`.
        """
        Param.__init__(self, name, lb=None, ub=None)

        if len(size) != 1:
            raise ValueError(f"Invalid dimensions")

        if value is None:
            value = np.zeros(size, dtype=FLOATX)

        self._init_type = "bias"
        self._init_value = value
        self.shared_var = aesara.shared(self.init_value, name=name, borrow=True)

    def __repr__(self):
        return f"Bias({self.name}, {self.shape})"

    @property
    def init_type(self):
        return self._init_type

    @property
    def T(self):
        return self.shared_var.T

    def __radd__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            b = aet.atleast_Nd(self(), n=other.ndim, left=False)
            return b + other

        else:
            return super().__radd__(other)

    def __add__(self, other):
        if isinstance(other, (TensorVariable, TensorSharedVariable)):
            b = aet.atleast_Nd(self(), n=other.ndim, left=False)
            return b + other

        else:
            return super().__add__(other)


class Weight(Param):
    def __init__(self, name, size, value=None, init_type=None):
        """Class object for neural net weight matrix

        Args:
            name (str): name of the parameter
            size (Union[tuple,list]): size of the array
            value (numpy.ndarray): initial values of the parameter. Defaults to `random.uniform(-0.1, 0.1, size)`
            init_type (str): initialization type, see notes

        Note:
            Initialization types are one of the following:

            * `"zeros"`: a 2-D array of zeros

            * `"he"`: initialization method for neural networks that takes into account
              the non-linearity of activation functions, e.g. ReLU or Softplus [^1]

            * `"glorot"`: initialization method that maintains the variance for
              symmetric activation functions, e.g. sigm, tanh [^2]

            [^1] He, K., Zhang, X., Ren, S. and Sun, J., 2015. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision (pp. 1026-1034).
            [^2] Glorot, X. and Bengio, Y., 2010, March. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics (pp. 249-256). JMLR Workshop and Conference Proceedings.

        !!! example
            Specifying a weight array:

            ```python
            code
            ```

        """
        from pycmtensor import config

        Param.__init__(self, name, lb=None, ub=None)

        # dimension of weight must be 2
        if len(size) != 2:
            raise ValueError(f"Invalid dimensions")

        rng = np.random.default_rng(seed=config.seed)
        n_in, n_out = size

        if value is None:
            if init_type == "zeros":
                value = np.zeros(size, dtype=FLOATX)
            elif init_type == "he":
                value = rng.normal(0, 1, size=size) * np.sqrt(2 / n_in)
            elif init_type == "glorot":
                scale = np.sqrt(6 / (n_in + n_out))
                value = rng.uniform(-1, 1, size) * scale
            elif init_type == "neg_glorot":
                scale = np.sqrt(6 / (n_in + n_out))
                value = rng.uniform(-1, 0, size) * scale
            else:
                debug(f"Weight {self.name} using default initialization U(-0.1, 0.1)")
                init_type = "uniform(-0.1, 0.1)"
                value = rng.uniform(-0.1, 0.1, size=size)

        self._init_type = init_type
        self._init_value = value
        self.shared_var = aesara.shared(self.init_value, name=name, borrow=True)

    @staticmethod
    def disp_init_types():
        return ["zeros", "he", "glorot", "neg_glorot"]

    @property
    def init_type(self):
        return self._init_type

    @property
    def T(self):
        return self.shared_var.T

    def __repr__(self):
        return f"Weight({self.name}, {self.shape})"
