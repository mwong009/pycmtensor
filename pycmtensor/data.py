# data.py
"""PyCMTensor data module"""
import aesara
import aesara.tensor as aet
import numpy as np
import pandas as pd

from pycmtensor import config, log

FLOATX = aesara.config.floatX


class Data:
    """Base Data class object"""

    def __init__(self, df: pd.DataFrame, choice: str):
        self.seed = config["seed"]
        self.split_frac = None
        self.k_fold = None
        self.config = config
        self.choice = choice
        self.scales = {}

        self.tensor = Variables(choice)
        self.pandas = PandasDataFrame(df, choice)

        for column in self.pandas.columns:
            if column == choice:
                self.tensor[column] = aet.ivector(column)
            else:
                self.tensor[column] = aet.vector(column)

            self.scales[column] = 1.0

    @property
    def x(self):
        return self.tensor.x

    @property
    def y(self):
        return self.tensor.y

    @property
    def all(self):
        return self.tensor.all

    def __getitem__(self, item: str):
        if item in [x.name for x in self.all]:
            return self.tensor[item]

    def split_db(self, split_frac=0.8):
        """Split database data into train and valid sets"""
        self.split_frac = split_frac
        self.pandas.split_pandas(self.seed, split_frac)

    def get_nrows(self) -> int:
        """Returns the lenth of the DataFrame object"""
        return len(self.pandas())

    def scale_data(self, **kwargs):
        """Scales data values by data/scale from ``kwargs:{column=scale}``"""
        for key, scale in kwargs.items():
            self.pandas[key] = self.pandas[key] / scale
            self.scales[key] *= scale
            log(10, f"Scaling {key} by {scale}")

    def autoscale_data(self, except_for=[None]):
        """Autoscale variable values to within -10.0 < x < 10.0

        Args:
                        except_for (list): list of str to skip autoscaling
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
                self.pandas[column] = self.pandas[column] / 10.0
                key = column
                scale = scale * 10.0
                max_val = np.max(np.abs(self.pandas[column]))
            # scaling complete
            log(10, f"Autoscaling {key} by {scale}")
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
    """Class object to store Pandas DataFrames"""

    def __init__(self, df: pd.DataFrame, choice: str):
        self.pandas = df
        if choice not in self.pandas.columns:
            raise ValueError(f"{choice} is not found in dataframe.")

        self.columns = self.pandas.columns

        # set default train and validation datasets
        self.train_dataset = [self.pandas]
        self.valid_dataset = [self.pandas]

    def __getitem__(self, item):
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
        self, tensors, index=None, batch_size=None, shift=None, split_type=None, k=0
    ) -> list[pd.DataFrame]:
        """Returns a list of DataFrame corresponding to the tensors input arg."""
        if split_type is None:
            dataset = self.pandas
        else:
            if split_type == "train":
                dataset = self.train_dataset[k]
            elif split_type == "valid":
                dataset = self.valid_dataset[k]
            else:
                raise ValueError(f'Valid arg for split:"train" or "valid"')

        datalist = []
        if index is None:
            datalist = [dataset[t.name] for t in tensors]
        else:
            start = index * batch_size + shift
            end = (index + 1) * batch_size + shift
            datalist = [dataset[t.name][start:end] for t in tensors]
        return datalist

    def split_pandas(self, seed, split_frac):
        df = self.pandas
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_train_samples = round(len(df) * split_frac)
        train_dataset = df.iloc[:n_train_samples, :].reset_index(drop=True)
        valid_dataset = df.iloc[n_train_samples + 1 :, :].reset_index(drop=True)
        self.train_dataset = [train_dataset]
        self.valid_dataset = [valid_dataset]


class Variables:
    """Class object to store TensorVariables"""

    def __init__(self, choice: str):
        self.variables = {}
        self.choice = choice

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
        """Returns all ``TensorVariable``"""
        return self.x + [self.y]
