# data.py
"""PyCMTensor data module"""
from typing import Literal, Union

import aesara
import aesara.tensor as aet
import numpy as np
import pandas as pd
from aesara.tensor.var import TensorVariable

from pycmtensor import config

from .logger import debug

__all__ = ["Data", "FLOATX"]


FLOATX = aesara.config.floatX


class Data:
    def __init__(self, df: pd.DataFrame, choice: str, **kwargs):
        """Base Data class object.

        Args:
            df (pandas.DataFrame): the input Pandas dataframe
            choice (str): column string name of the choice dependent variable
            **kwargs: Keyword arguments, accepted arguments are `drop:pd.Series`,
                `autoscale:bool`, `autoscale_except:list[str]`, `split:float`

        Attributes:
            x (list[TensorVariable]): list of tensors corresponding to input features
            y (list[TensorVariable]): list of the dependent choice variable tensor
            all (list[TensorVariable]): combined list of x and y variables

        Note:
            The following is an example initialization of the swissmetro dataset::

                swissmetro = pd.read_csv(\"../data/swissmetro.dat\", sep=\"\\t\")
                db = pycmtensor.Data(
                    df=swissmetro,
                    choice=\"CHOICE\",
                    drop=[swissmetro[\"CHOICE\"]==0],
                    autoscale=True,
                    autoscale_except=[\"ID\", \"ORIGIN\", \"DEST\"],
                    split=0.8,
                )
        """
        self.seed = config["seed"]
        self.split_frac = None
        self.k_fold = None
        self.config = config
        self.choice = choice
        self.scales = {}

        # drop rows for "drop" in kwargs
        if "drop" in kwargs:
            for d in kwargs["drop"]:
                df.drop(df[d].index, inplace=True)

        # reindex choices to start from index-0
        if df[choice].min() != 0:
            df[choice] -= df[choice].min()

        # prepare tensor and pandas data
        self.pandas = PandasDataFrame(df, choice)
        self.tensor = Variables(choice, self.pandas.columns)
        self.scales = {column: 1.0 for column in self.pandas.columns}

        # autoscale data if argument is set
        if "autoscale" in kwargs:
            ex = None
            if "autoscale_except" in kwargs:
                ex = kwargs["autoscale_except"]
            self.autoscale_data(except_for=ex)

        if "split" in kwargs:
            self.split_db(split_frac=kwargs["split"])

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
    def n_train_samples(self):
        return len(self.pandas.train_dataset[0])

    @property
    def n_valid_samples(self):
        return len(self.pandas.valid_dataset[0])

    @property
    def train_data(self):
        return self.pandas.inputs(self.all, split_type="train")

    @property
    def valid_data(self):
        return self.pandas.inputs(self.all, split_type="valid")

    def __getitem__(self, item: Union[str, list]) -> TensorVariable:
        if isinstance(item, list):
            return [self.tensor[x.name] for x in self.all if x.name in item]
        if item in [x.name for x in self.all]:
            return self.tensor[item]
        else:
            raise ValueError(f"{item} not a valid Variable name")

    def split_db(self, split_frac: float):
        """Split database data into train and valid sets

        Arg:
            split_frac (float): fractional value between 0.0 and 1.0.
        """
        self.split_frac = split_frac
        self.pandas.split_pandas(self.seed, split_frac)

    def get_nrows(self) -> int:
        """Returns the lenth of the DataFrame object"""
        return len(self.pandas())

    def get_train_data(self, tensors, index=None, batch_size=None, shift=None):
        """Alias to get train data slice from `self.pandas.inputs()`

        See :meth:`PandasDataFrame.inputs()` for details
        """
        return self.pandas.inputs(tensors, index, batch_size, shift, "train")

    def get_valid_data(self, tensors, index=None, batch_size=None, shift=None):
        """Alias to get valid data slice from `self.pandas.inputs()`

        See :meth:`PandasDataFrame.inputs()` for details
        """
        return self.pandas.inputs(tensors, index, batch_size, shift, "valid")

    def scale_data(self, **kwargs):
        """Scales data values by data/scale from `key=scale` keyword argument

        Args:
            **kwargs: {key: scale} keyword arguments
        """
        for key, scale in kwargs.items():
            self.pandas[key] = self.pandas[key] / scale
            self.scales[key] *= scale
            debug(f"Scaling {key} by {scale}")

    def autoscale_data(self, except_for=[None]):
        """Autoscale variable values to within -10.0 < x < 10.0

        Args:
            except_for (list[str]): list of column labels to skip autoscaling step
        """
        x_columns = [x.name for x in self.x]
        if type(except_for) != type([]):
            except_for = [except_for]
        for column in self.pandas.columns:
            if (column in except_for) or (column not in x_columns):
                continue
            max_val = np.max(np.abs(self.pandas[column]))
            if max_val <= 10:
                continue
            scale = 1.0
            while max_val > 10:
                self.scale_data(**{column: 10.0})
                scale = scale * 10.0
                key = column
                max_val = np.max(np.abs(self.pandas[column]))

            self.scales[key] = scale

    def info(self):
        """Outputs information about the Data class object"""
        msg = (
            f"choice = {self.choice}\n"
            f"nrows = {self.get_nrows()}\n"
            f"x = {self.x}\n"
            f"y = {self.y}\n"
            f"split_frac = {self.split_frac}\n"
        )
        return msg


class PandasDataFrame:
    def __init__(self, df: pd.DataFrame, choice: str):
        """Class object to store Pandas DataFrame.

        Args:
            df (pandas.DataFrame): the input Pandas dataframe
            choice (str): column string name of the choice dependent variable
        """
        self.pandas = df
        if choice not in self.pandas.columns:
            raise ValueError(f"{choice} is not found in dataframe.")

        self.columns = self.pandas.columns

        # set default train and validation datasets
        self.train_dataset = [self.pandas]
        self.valid_dataset = [self.pandas]

    def __getitem__(self, item):
        if isinstance(item, list):
            for i in item:
                if i not in self.pandas.columns:
                    raise ValueError(f"{item} not in PandasDataFrame class.")
        else:
            if item not in self.pandas.columns:
                raise ValueError(f"{item} not in PandasDataFrame class.")
        return self.pandas[item]

    def __setitem__(self, item: str, value):
        if item not in self.pandas.columns:
            raise ValueError(f"{item} not in PandasDataFrame class.")
        self.pandas[item] = value

    def __getattr__(self, attr):
        if attr not in self.pandas.columns:
            raise ValueError(f"{attr} not in PandasDataFrame class.")
        return self.pandas[attr]

    def __call__(self):
        return self.pandas

    def inputs(
        self,
        tensors: list[TensorVariable],
        index: int = None,
        batch_size: int = None,
        shift: int = 0,
        split_type: Literal["train", "valid"] = "train",
    ) -> list[pd.DataFrame]:
        """Returns a list of DataFrame corresponding to the tensors input

        Args:
            tensors (list[TensorVariable]): list of tensors as an index to call the
                pandas dataset
            index (int, optional): starting index of the return dataset slice. Defaults
                to `None` and returns the entire dataset.
            batch_size (int, optional): dataset slice length. Defaults to maximum
                length of the dataset.
            shift (int, optional): index offset. Defaults to 0.
            split_type (str, optional): {'train', 'valid'}
                Defines the specific split of the dataset to return. Possible values
                are `train` or `valid`. If `self.split_pandas()` was not called, both
                `train` or `valid` arguments return the same dataset.

        """
        if split_type == "train":
            dataset = self.train_dataset[0]
        elif split_type == "valid":
            dataset = self.valid_dataset[0]
        else:
            raise ValueError(f"Valid arg {split_type} for split_type")

        datalist = []
        if index is None:
            datalist = [dataset[t.name] for t in tensors]
        else:
            if batch_size is None:
                batch_size = len(dataset)
            start = index * batch_size + min(batch_size, shift)
            end = (index + 1) * batch_size + min(batch_size, shift)
            datalist = [dataset[t.name].iloc[start:end] for t in tensors]
        return datalist

    def split_pandas(self, seed: int, split_frac: float):
        """Function to split the pandas dataset into train and valid splits

        Args:
            seed (int): random seed value
            split_frac (float): fractional value between 0.0 and 1.0
        """
        df = self.pandas
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_train_samples = round(len(df) * split_frac)
        train_dataset = df.iloc[:n_train_samples, :].reset_index(drop=True)
        valid_dataset = df.iloc[n_train_samples + 1 :, :].reset_index(drop=True)
        self.train_dataset = [train_dataset]
        self.valid_dataset = [valid_dataset]


class Variables:
    def __init__(self, choice: str, columns: list[str]):
        """Class object to store `TensorVariable`.

        Args:
            choice (str): column string label of the choice dependent variable
            columns (list[str]): list of pandas column labels

        Attributes:
            x (list[TensorVariable]): list of tensors corresponding to input features
            y (list[TensorVariable]): list of the dependent choice variable tensor
            all (list[TensorVariable]): combined list of x and y variables
        """
        self.variables = {}
        self.choice = choice

        for column in columns:
            if column == choice:
                self[column] = aet.ivector(column)

            else:
                self[column] = aet.vector(column)

    def __getitem__(self, item: str):
        if item not in self.variables:
            raise ValueError(f"{item} does not exist in Variables class.")
        return self.variables[item]

    def __setitem__(self, key: str, value: aet.TensorVariable):
        if not type(value) == aet.TensorVariable:
            raise TypeError(f"{value} must be a aet.TensorVariable type object.")
        if key != self.choice:
            self.variables[key] = value
        else:
            self.choice = value

    @property
    def x(self) -> list[aet.TensorVariable]:
        """Returns only the x ``aet.TensorVariable`` of the class"""
        x_tensors = [x for _, x in self.variables.items()]
        return x_tensors

    @property
    def y(self) -> aet.TensorVariable:
        """Returns only the y ``aet.TensorVariable`` of the class"""
        y_tensor = self.choice
        if type(y_tensor) == str:
            raise ValueError(f"Choice variable not set yet.")
        return y_tensor

    @property
    def all(self) -> list[aet.TensorVariable]:
        """Returns all ``TensorVariable`` objects"""
        return self.x + [self.y]
