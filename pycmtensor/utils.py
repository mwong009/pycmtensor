"""PyCMTensor utils module

This module provides utility functions for formatting time and numbers.

Functions:
- time_format(seconds): Converts a number of seconds into a formatted string representing the time in hours, minutes, and seconds.
- human_format(number): Converts a number into a human-readable format with a magnitude suffix.

Example Usage:
```python
time_str = time_format(3661)
print(time_str)  # Output: '01:01:01'

number_str = human_format(1234567890)
print(number_str)  # Output: '1.23B'
```
"""

suffixes = ["", "K", "M", "B", "T", "P"]


def time_format(seconds):
    """
    Converts a number of seconds into a formatted string representing the time in hours, minutes, and seconds.

    Args:
        seconds (int or float): The number of seconds.

    Returns:
        str: The formatted time string in the format 'HH:MM:SS'.
    """
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def human_format(number):
    """
    Converts a number into a human-readable format with a magnitude suffix.

    Args:
        number (int or float): The number to be converted.

    Returns:
        str: The formatted number string with a magnitude suffix.
    """
    number = float(f"{number:.3g}")
    magnitude = 0
    while abs(number) >= 1000:
        magnitude += 1
        number /= 1000.0
    return (
        f"{number}".rstrip("0").rstrip(".") + ["", "K", "M", "B", "T", "P"][magnitude]
    )
