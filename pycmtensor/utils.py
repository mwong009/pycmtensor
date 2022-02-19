# utils.py


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
