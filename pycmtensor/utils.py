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


def learn_rate_tempering(iter, patience, lr_init):
    """Tempering learning rate as a function of iteration step.

    Args:
        iter (int): Iteration step.
        patience (int): The maximum number of iterations.
        lr_init (float): Initial learning rate.

    Returns:
        float: A new learning rate value
    """
    if (iter / patience) < 0.65:
        if (iter / patience) < 0.2:
            lr = lr_init
        else:
            lr = lr_init / 5.0
    else:
        lr = lr_init / 10.0

    return lr
