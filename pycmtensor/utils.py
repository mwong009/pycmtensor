# utils.py
"""PyCMTensor utils module"""
import dill as pickle


def save_to_pickle(model):
    with open(model.name + ".pkl", "wb") as f:
        pickle.dump(model, f)  # save model to pickle


def time_format(seconds):
    minutes, seconds = divmod(round(seconds), 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
    else:
        hours = 0
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
