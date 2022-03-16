# utils.py
"""Collection of useful functions and methods"""


def tqdm_nb_check(notebook: bool):
    """Check if `__main__` is running from a notebook or not.

    Args:
        notebook (book): Boolean flag from `train{}` if program is running in a Jupyter Notebook.

    Returns:
        `tqdm`: Returns the tqdm module if not running in a notebook, else returns a tqdm.notebook module.
    """
    if not isinstance(notebook, bool):
        raise TypeError(f"{notebook} is not a {bool.__name__} type")
    if notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    return tqdm


def save_to_pickle(model):
    import dill as pickle

    with open(model.name + ".pkl", "wb") as f:
        pickle.dump(model, f)  # save model to pickle
