import pandas as pd
from attr import has


class Tracker:
    """Base class for Tracker objects."""

    def __init__(self):
        pass


class IterationTracker(Tracker):
    def __init__(self, name="IterationTracker"):
        super().__init__()
        self.name = name
        self.columns = ["iteration"]
        self._data = pd.DataFrame()
        self._data.index.name = "iteration"

    def add(self, iter, key, value):
        if not hasattr(self, key):
            setattr(self, key, [])
            self.columns.append(key)
        k = getattr(self, key)
        k.append((iter, value))

    def get_data(self):
        concat_list = []
        for column in self.columns[1:]:
            d = getattr(self, column)
            df = pd.DataFrame(d, columns=["iteration", column]).set_index("iteration")
            df[column] = df[column].astype("float64")
            concat_list.append(df)
        data = pd.concat(concat_list, axis=1)
        return data
