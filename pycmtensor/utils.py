# utils.py


def tqdm_nb_check(notebook):
    assert isinstance(notebook, bool)
    if notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    return tqdm


def learn_rate_tempering(iter, patience, lr_init):
    """Tempering learning rate as a function of iteration step

    Args:
        iter (int): Iteration step.
        patience (int): The maximum number of iterations.
        lr_init (float): Initial learning rate

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
