# utils.py
"""Collection of useful functions and methods"""

from pycmtensor import logger as log

from .logger import PyCMTensorError
from .models import PyCMTensorModel


def save_to_pickle(model):
    import dill as pickle

    with open(model.name + ".pkl", "wb") as f:
        pickle.dump(model, f)  # save model to pickle


def inspect_model(model):
    """Raises and error if `model` is not a valid ``PyCMTensorModel`` class.

    Args:
        model (PyCMTensorModel): the constructed model class.

    Raises:
        PyCMTensorError: logs an error if the model class is an invalid class.

    Returns:
        PyCMTensorModel: Returns the ``model`` object.

    Example:
        .. code-block :: python

            import pycmtensor as cmt
            from pycmtensor.models import MNLModel
            db = cmt.Database(pandasDatabase=some_pandas_data)
            ...

            model = MNLogit(u=U, av=AV, database=db, name="mymodel")
            inpect_model(model)

    """
    if not isinstance(model, PyCMTensorModel):
        msg = f"{model} is not a valid PyCMTensorModel model."
        log.error(msg)
        raise PyCMTensorError(msg)
    return model
