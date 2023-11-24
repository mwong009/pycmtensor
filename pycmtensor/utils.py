# utils.py
"""PyCMTensor utils module"""


def time_format(seconds):
    minutes, seconds = divmod(round(seconds), 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
    else:
        hours = 0
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def human_format(number):
    number = float(f"{number:.3g}")
    magnitude = 0
    while abs(number) >= 1000:
        magnitude += 1
        number /= 1000.0
    return (
        f"{number}".rstrip("0").rstrip(".") + ["", "K", "M", "B", "T", "P"][magnitude]
    )
