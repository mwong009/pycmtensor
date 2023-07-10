# dataset.py
# converts pandas dataframe into an xarray dataset

import aesara.tensor as aet
import pandas as pd

from pycmtensor import config

from .logger import debug, info

__all__ = ["Dataset"]


class Dataset:
    def __init__(self, df: pd.DataFrame, choice: str):
        """Base PyCMTensor Dataset class object.

        Args:
            df (pandas.DataFrame):
            choice (str):

        Attributes:
            n (int):
            x (list):
            y (TensorVariable):
            scale (dict):
            choice (str):
            ds (dict)
            split_frac (float):
            train_index (list):
            valid_index (list):
            n_train (int):
            n_valid (int):

        Example:
            Example initalization of a pandas dataset:

            ```python
            ds = Dataset(
                df=pd.read_csv("datafile.csv", sep=","),
                choice="mode",
                alts={0: "car", 1:"bus", 2:"train"}
            )
            ds.split(frac=0.8)
            ```

        """
        if choice not in df.columns:
            raise IndexError(f"{choice} not found in dataframe.")

        df[choice] = df[choice].astype("int")  # ensure choice variable is an integer
        df.reset_index(drop=True, inplace=True)
        while df[choice].min() > 0:
            df[choice] -= df[choice].min()

        self.index = df.index.values

        self.n = len(df)
        self.x = []
        for name in list(df.columns):
            if name == choice:
                self.y = aet.ivector(name)
            else:
                self.x.append(aet.vector(name))

        df = df.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)

        ds = {}
        for name in list(df.columns):
            ds[name] = df[name].values

        self.scale = {var.name: 1.0 for var in self.x}
        self.choice = choice
        self.ds = ds
        self.split_frac = 1.0

    def __call__(self):
        return self.ds

    def __getitem__(self, key):
        if key in [var.name for var in self.x]:
            i = [x.name for x in self.x].index(key)
            return self.x[i]
        if key == self.y.name:
            return self.y
        else:
            raise KeyError

    @property
    def n_train(self) -> int:
        return len(self.train_index)

    @property
    def n_valid(self) -> int:
        return len(self.valid_index)

    @property
    def train_index(self) -> list:
        n = round(self.n * self.split_frac)
        return self.index[:n]

    @property
    def valid_index(self) -> list:
        n = round(self.n * self.split_frac)
        return self.index[n:]

    def drop(self, variables: list):
        """TODO"""
        for variable in variables:
            if (variable in self.ds) and (variable != self.choice):
                i = [x.name for x in self.x].index(variable)
                del self.x[i]
                del self.scale[variable]
                debug(f"Dropped input variable '{variable}' from dataset")

            else:
                raise KeyError

    def scale_variable(self, variable, factor):
        """Multiply values of the variable by factor 1/factor.

        Args:
            variable (str): the name of the variable or a list of variable names
            factor (float): the scaling factor
        """
        self.ds[variable] = self.ds[variable] / factor
        self.scale[variable] = self.scale[variable] * factor

    def split(self, frac):
        """TODO"""
        n = round(self.n * frac)
        self.split_frac = frac
        info(f"n_train_samples:{self.n_train} n_valid_samples:{self.n_valid}")

    def _dataset_slice(self, tensors, index, batch_size, shift, n_index):
        """Internal method call for self.train_dataset or self.valid_dataset

        Args:
            tensors (TensorVariable): tensor or list of tensors
            index (int):
            batch_size (int):
            shift (int):
            n_index (list): list of index values of the [train|valid] dataset
        """
        if not isinstance(tensors, list):
            tensors = [tensors]

        tensor_names = [t.name for t in tensors]
        for name in tensor_names:
            if name not in list(self.ds):
                raise KeyError(f"{name} not in dataset. {list(self.ds)}")

        if index is None:
            i = n_index
        else:
            if batch_size is None:
                batch_size = len(n_index)
            if shift is None:
                shift = 0
            start = index * batch_size + min(batch_size, shift)
            end = (index + 1) * batch_size + min(batch_size, shift)
            i = n_index[start:end]

        _ds = [self.ds[name][i] for name in tensor_names]

        return _ds

    def train_dataset(self, variables, index=None, batch_size=None, shift=None):
        """Return a slice of the training dataset with the sequence matching the list of variables"""
        n_index = self.train_index
        return self._dataset_slice(variables, index, batch_size, shift, n_index)

    def valid_dataset(self, variables, index=None, batch_size=None, shift=None):
        """Return a slice of the valid dataset with the sequence matching the list of variables"""
        n_index = self.valid_index

        return self._dataset_slice(variables, index, batch_size, shift, n_index)
