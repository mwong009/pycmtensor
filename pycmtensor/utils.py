# utils.py
"""Collection of useful functions and methods"""


def save_to_pickle(model):
    import dill as pickle

    with open(model.name + ".pkl", "wb") as f:
        pickle.dump(model, f)  # save model to pickle
