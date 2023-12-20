# regularizers.py
"""PyCMTensor regularizers module

This module contains regularizer terms for use in the cost function
"""

import aesara.tensor as aet
import numpy as np

__all__ = ["Regularizers"]


class Regularizers:
    @staticmethod
    def l1_positive(params, weight=0.01):
        """Compute the L1 norm of positive valued tensors.

        This function calculates the L1 norm of the given tensors `params` by summing
        the absolute values of its elements. Only the positive values are considered
        in the calculation.

        Explanation: This function penalizes the L1 norm of only the positive values of the params

        Args:
            params (Union[list[TensorVariable], TensorVariable]): The tensors for which the L1 norm is to be computed.
            weight (float, optional): A weight factor to scale the L1 norm. Defaults to 0.01.

        Returns:
            float: The L1 norm of the positive valued tensor.
        """
        if not isinstance(params, list):
            params = [params]
        return weight * aet.sum([aet.sum(aet.clip(p(), 0, np.inf)) for p in params])

    @staticmethod
    def l1(params, weight=0.01):
        """Compute the L1 norm of the tensors.

        This function calculates the L1 norm of the given tensors by summing the absolute values of their elements.

        Args:
            params (Union[list[TensorVariable], TensorVariable]): The parameters to compute the L1 norm.
            weight (float): The value for penalizing the regularization term. Defaults to 0.01.

        Returns:
            (TensorVariable): The L1 norm, which is the sum of absolute values of the tensors.
        """
        if not isinstance(params, list):
            params = [params]

        return weight * aet.sum([aet.sum(aet.abs(p())) for p in params])

    @staticmethod
    def l2(params, weight=0.001):
        """Compute the L2 norm of the tensors.

        This function calculates the L2 norm of the given tensors. It takes a list of tensors or a single tensor as input and applies L2 regularization to penalize large parameter values.

        Args:
            params (Union[list[TensorVariable], TensorVariable]): The tensors to calculate the L2 norm for.
            weight (float, optional): The weight for penalizing the regularization term. Defaults to 0.001.

        Returns:
            float: The L2 norm of the tensors multiplied by the weight.
        """
        if not isinstance(params, list):
            params = [params]

        return weight * aet.sum([aet.sum(aet.square(p())) for p in params])
