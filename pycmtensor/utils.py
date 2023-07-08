# utils.py
"""PyCMTensor utils module"""


def time_format(seconds):
    minutes, seconds = divmod(round(seconds), 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
    else:
        hours = 0
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
