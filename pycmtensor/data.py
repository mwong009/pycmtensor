# data.py
"""PyCMTensor data module"""
from typing import Literal, Union

import aesara
import aesara.tensor as aet
import numpy as np
import pandas as pd
from aesara.tensor.var import TensorVariable

from .logger import debug

__all__ = ["Data", "FLOATX"]


FLOATX = aesara.config.floatX


class Data:
    def __init__(self, df: pd.DataFrame, choice: str, split=None):
        """Base Data class object.

        Args:
            df (pandas.DataFrame): the input Pandas dataframe
            choice (str): column string name of the choice dependent variable
            split: `float` A fraction of the dataset as training dataset (the remainer is used as validation dataset). Defaults to `None`.

        Attributes:
            Data.choice (str): the choice variable name
            Data.columns (list): list of columns in the dataframe
            Data.x (list(TensorVariable)): A list of tensor variables for the model input
            Data.y (TensorVariable): The tensor variable for the model output
            Data.all (list(TensorVariable)): A list of all tensor variables `Data.x + [Data.y]`
            Data.n_train (int): Number of training samples
            Data.n_valid (int): Number of validation samples
            Data.n_rows (int): Number of samples in the dataset

        Note:
            The following is an example initialization of the swissmetro dataset::

                swissmetro = pd.read_csv(\"../data/swissmetro.dat\", sep=\"\\t\")
                db = pycmtensor.Data(
                    df=swissmetro,
                    choice=\"CHOICE\",
                    split=0.8,
                )
        """
        self.split = split
        self.seed = 42069

        # scale factor for each column
        self.scales = {column: 1.0 for column in df.columns}

        # check if choice is valid
        assert choice in df.columns

        # re-index choices to start from index-0
        assert str(df[choice].dtype).startswith("int")
        while df[choice].min() > 0:
            df[choice] -= df[choice].min()

        # prepare tensor and pandas data
        self.pandas = PandasDataFrame(df, choice)
        self.tensor = Variables(choice, df.columns)
        self.pandas.split(self.split, self.seed)

    @property
    def choice(self):
        return self.pandas.choice

    @property
    def x(self):
        return self.tensor.x

    @property
    def y(self):
        return self.tensor.y

    @property
    def all(self):
        return self.tensor.all

    @property
    def n_train(self):
        return len(self.pandas.train_dataset)

    @property
    def n_valid(self):
        return len(self.pandas.valid_dataset)

    @property
    def n_rows(self):
        return len(self.pandas())

    def __getitem__(self, item):
        if item == self.choice:
            return self.tensor.choice
        else:
            if isinstance(item, list):
                return [self.tensor[i] for i in item]
            else:
                return self.tensor[item]

    def train(
        self, tensors=None, index=None, batch_size=None, shift=None, numpy_out=False
    ):
        """Alias to get train data slice from `self.pandas.train()`.

        See :meth:`PandasDataFrame.train()` for details.
        """
        if tensors is None:
            tensors = self.all
        return self.pandas.train(tensors, index, batch_size, shift, numpy_out)

    def valid(
        self, tensors=None, index=None, batch_size=None, shift=0, numpy_out=False
    ):
        """Alias to get valid data slice from `self.pandas.valid()`.

        See :meth:`PandasDataFrame.valid()` for details.
        """
        if tensors is None:
            tensors = self.all
        return self.pandas.valid(tensors, index, batch_size, shift, numpy_out)

    def scale(self, variable, factor):
        """Scales the values of `variable` by `factor`.

        Args:
            variable (str): name of the variable
            factor (float): the scaling factor `n` applied to each value `x/n`
        """
        self.pandas[variable] = self.pandas[variable] / factor
        self.scales[variable] *= factor

    def show_info(self):
        """Outputs information about the Data class object"""
        msg = (
            f"choice = {self.choice}\n"
            f"n_rows = {self.n_rows}\n"
            f"n_train = {self.n_train}\n"
            f"n_valid = {self.n_valid}\n"
            f"split = {self.split}\n"
            f"y = {self.y}\n"
            f"x = {self.x}\n"
        )
        print(msg)


class PandasDataFrame:
    def __init__(self, df: pd.DataFrame, choice: str):
        """Class object to store Pandas DataFrame.

        Args:
            df (pandas.DataFrame): the input Pandas dataframe
            choice (str): column string name of the choice dependent variable
        """
        self.pandas = df
        self.choice = choice

        # set default train and validation datasets
        self.train_dataset = self.valid_dataset = self.pandas

    def __getitem__(self, item):
        return self.pandas[item]

    def __setitem__(self, item: str, value):
        self.pandas[item] = value

    def __getattr__(self, attr):
        return self.pandas[attr]

    def __call__(self):
        return self.pandas

    def train(self, tensors, index=None, batch_size=None, shift=None, numpy_out=False):
        """Return a slice of the training dataset with the same sequence as tensors.

        Args:
            tensors (TensorVariable): A tensor variable or a list of tensor variables
            index (int, optional): starting index of the dataset. Defaults to None. If None is given, returns the entire dataset.
            batch_size (int, optional): dataset slice length. Defaults to None.
            shift (int, optional): index offset. Defaults to None.
            numpy_out (bool, optional): If True, returns a list of numpy array objects.
        """
        df = self.train_dataset
        if not isinstance(tensors, list):
            tensors = [tensors]

        return self._inputs(df, tensors, index, batch_size, shift, numpy_out)

    def valid(self, tensors, index=None, batch_size=None, shift=None, numpy_out=False):
        """Return a slice of the valid dataset with the same sequence as tensors.

        Args:
            tensors (TensorVariable): A tensor variable or a list of tensor variables
            index (int, optional): starting index of the dataset. Defaults to None. If None is given, returns the entire dataset.
            batch_size (int, optional): dataset slice length. Defaults to None.
            shift (int, optional): index offset. Defaults to None.
            numpy_out (bool, optional): If True, returns a list of numpy array objects.
        """
        df = self.valid_dataset
        if not isinstance(tensors, list):
            tensors = [tensors]

        return self._inputs(df, tensors, index, batch_size, shift, numpy_out)

    def _inputs(self, df, tensors, index, batch_size, shift, numpy_out):
        """Returns a list of DataFrame or numpy array corresponding to the tensors input. Internal use only."""
        df_new = []
        if index is None:
            df_new = [df[t.name] for t in tensors]
        else:
            if batch_size is None:
                batch_size = len(df)
            if shift is None:
                shift = 0
            start = index * batch_size + min(batch_size, shift)
            end = (index + 1) * batch_size + min(batch_size, shift)
            df_new = [df[t.name].iloc[start:end] for t in tensors]

        if numpy_out:
            df_new = [d.values for d in df_new]
        return df_new

    def split(self, frac, seed=None):
        """Function to split the pandas dataset into train and valid datasets.

        Args:
            frac (float): fractional value between 0.0 and 1.0
            seed (int): random seed value. Defaults to None
        """
        n = 0
        df = self.pandas
        if frac is not None:
            df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            n = round(len(df) * frac)
            self.train_dataset = df.iloc[:n, :].reset_index(drop=True)
            self.valid_dataset = df.iloc[n + 1 :, :].reset_index(drop=True)
        return f"split -> training samples={n}, valid samples={len(df)-n}"


class Variables:
    def __init__(self, choice, columns):
        """Class object to store `TensorVariable`.

        Args:
            choice (str): the name of the choice variable
            columns (list[str]): list of pandas column labels

        Attributes:
            Variables.x (list(TensorVariable)): A list of tensor variables for the model input
            Variables.y (TensorVariable): The tensor variable for the model output
            Variables.all (list(TensorVariable)): A list of all tensor variables `Data.x + [Data.y]`
        """
        self.variables = {}
        for column in columns:
            if column == choice:
                self.choice = aet.ivector(column)
            else:
                self.variables[column] = aet.vector(column)

    def __getitem__(self, item):
        return self.variables[item]

    @property
    def x(self):
        x_tensors = [x for _, x in self.variables.items()]
        return x_tensors

    @property
    def y(self):
        return self.choice

    @property
    def all(self) -> list[aet.TensorVariable]:
        """Returns all ``TensorVariable`` objects"""
        return self.x + [self.y]
