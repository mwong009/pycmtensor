# regularizers.py
"""PyCMTensor regularizers module

This module contains regularizer terms for use in the cost function
"""

import aesara.tensor as aet

__all__ = ["Regularizers"]


class Regularizers(object):
    def __init__(self):
        pass

    @staticmethod
    def l1(params, weight=0.001):
        """compute the L1 norm of the tensors

        Args:
            params (Union[list[TensorVariable], TensorVariable]): the parameters to compute the L1 norm
            weight (float): value for penalizing the regularization term

        Returns:
            (TensorVariable): the L1 norm (sum of absolute values)
        """
        if not isinstance(params, list):
            params = [params]

        return weight * aet.sum([aet.sum(aet.abs(p())) for p in params])

    @staticmethod
    def l2(params, weight=0.0001):
        """compute the L2 norm of the tensors

        Args:
            params (Union[list[TensorVariable], TensorVariable]): the parameters to compute the L2 norm
            weight (float): value for penalizing the regularization term

        Returns:
            (TensorVariable): the L2 norm (sum of squared values)
        """
        if not isinstance(params, list):
            params = [params]

        return weight * aet.sum([aet.sum(aet.square(p())) for p in params])
