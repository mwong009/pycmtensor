import numpy as np
import pandas as pd
from attr import has


class Tracker:
    """Base class for Tracker objects."""

    def __init__(self):
        pass


class IterationTracker(Tracker):
    def __init__(self, iterations, name="IterationTracker"):
        super().__init__()
        self.name = name
        self.iterations = iterations
        self.columns = []
        self._data = pd.DataFrame()
        self._data.index.name = "iteration"
        self._iteration_tracker = 0

    def add(self, i, key, value):
        if not hasattr(self, key):
            setattr(self, key, np.zeros((self.iterations,)))
            self.columns.append(key)
        k = getattr(self, key)
        k[i] = value
        self._iteration_tracker = i

    def get_data(self):
        concat_list = []
        for column in self.columns:
            d = getattr(self, column)
            df = pd.DataFrame(d, columns=[column])
            df[column] = df[column].astype("float64")
            concat_list.append(df)
        data = pd.concat(concat_list, axis=1)
        data = self.trim(data)
        return data

    def trim(self, data):
        data = data.iloc[: self._iteration_tracker, :]
        return data
