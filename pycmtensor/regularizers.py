# regularizers.py
"""PyCMTensor regularizers module

This module contains regularizer terms for use in the cost function
"""

import aesara.tensor as aet

__all__ = ["Regularizers"]


class Regularizers:
    @staticmethod
    def l1(params, weight=0.001):
        """compute the L1 norm of the tensors

        Args:
            params (Union[list[TensorVariable], TensorVariable]): The parameters to compute the L1 norm.
            weight (float): The value for penalizing the regularization term. Default value is 0.001.

        Returns:
            (TensorVariable): The L1 norm, which is the sum of absolute values of the tensors.
        """
        if not isinstance(params, list):
            params = [params]

        return weight * aet.sum([aet.sum(aet.abs(p())) for p in params])

    @staticmethod
    def l2(params, weight=0.0001):
        """Compute the L2 norm of the tensors

        Args:
            params (Union[list[TensorVariable], TensorVariable]): The parameters to compute the L2 norm.
            weight (float): The value for penalizing the regularization term. Default value is 0.0001.

        Returns:
            (TensorVariable): The L2 norm, which is the sum of squared values of the tensors.
        """
        if not isinstance(params, list):
            params = [params]

        return weight * aet.sum([aet.sum(aet.square(p())) for p in params])
