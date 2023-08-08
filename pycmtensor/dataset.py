# dataset.py
# converts pandas dataframe into an xarray dataset

from typing import Union

import aesara.tensor as aet
from aesara.tensor.var import TensorVariable

from pycmtensor import config

from .logger import debug, info

__all__ = ["Dataset"]


class Dataset:
    def __init__(self, df, choice):
        """Base PyCMTensor Dataset class object

        This class stores the data in an array format, and a symbolic tensor reference
        variable object. To call the tensor variable, we invoke the label of the
        variable as an item in the Dataset class, like so:
        ```python
        ds = Dataset(df=df, choice="choice")
        return ds["label_of_variable"]  -> TensorVariable
        ```

        To call the data array, we use the `train_dataset()` or `valid_dataset()`
        method. See method reference for info about the arguments. For example:
        ```python
        # to get the data array for variable "time"
        arr = ds.train_dataset(ds["time"])
        ```

        Args:
            df (pandas.DataFrame): the pandas dataframe object to load
            choice (str): the name of the choice variable

        Attributes:
            n (int): total number of rows in the dataset
            x (list[TensorVariable]): the full list of (input) `TensorVariable` objects
                to build the tensor expression from
            y (TensorVariable): the output (choice) `TensorVariable` object
            scale (dict): a dictionary of `float` values to store the scaling factor
                used for each variable
            choice (str): the name of the choice variable
            ds (dict): a dictionary of `numpy.ndarray` to store the values of each
                variable
            split_frac (float): the factor used to split the dataset into training and
                validation datasets
            train_index (list): the list of values of the indices of the training
                dataset
            valid_index (list): the list of values of the indices of the validation
                dataset
            n_train (int): the size of the training dataset
            n_valid (int): the size of the validation dataset

        Example:
            Example initalization of a pandas dataset:

            ```python
            ds = Dataset(df=pd.read_csv("datafile.csv", sep=","), choice="mode")
            ds.split(frac=0.8)
            ```

            Attributes can be access by invoking:
            ```python
            print(ds.choice)
            ```

            Output:
            ```bash
            'car'
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
        if isinstance(key, (list, tuple)):
            return self._make_tensor(key)
        else:
            if key in [var.name for var in self.x]:
                i = [x.name for x in self.x].index(key)
                return self.x[i]
            if key == self.y.name:
                return self.y
            else:
                raise KeyError

    def _make_tensor(self, keys):
        # if tensor inputs are list of strings, convert them to tensors
        if all(isinstance(k, str) for k in keys):
            keys = [self[k] for k in keys]
        else:
            raise TypeError(f"Multiple types found in {keys}.")
        return aet.as_tensor_variable(keys)

    @property
    def n_train(self) -> int:
        return len(self.train_index)

    @property
    def n_valid(self) -> int:
        return len(self.valid_index)

    @property
    def train_index(self) -> list:
        if self.split_frac == 1:
            return self.index

        n = round(self.n * self.split_frac)
        return self.index[:n]

    @property
    def valid_index(self) -> list:
        if self.split_frac == 1:
            return self.index
        n = round(self.n * self.split_frac)
        return self.index[n:]

    def drop(self, variables):
        """Method for dropping `variables` from the dataset

        Args:
            variables (list[str]): list of `str` variables from the dataset to drop

        Raises:
            KeyError: raises an error if any item in `variables` is not found in the dataset or item is the choice variable

        !!! Warning
            Choice variable cannot be explicity dropped.
        """
        for variable in variables:
            if (variable in self.ds) and (variable != self.choice):
                i = [x.name for x in self.x].index(variable)
                del self.x[i]
                del self.scale[variable]
                del self.ds[variable]
                debug(f"Dropped input variable '{variable}' from dataset")

            else:
                raise KeyError

    def scale_variable(self, variable, factor):
        """Multiply values of the `variable` by $1/\\textrm{factor}$.

        Args:
            variable (str): the name of the variable or a list of variable names
            factor (float): the scaling factor
        """
        self.ds[variable] = self.ds[variable] / factor
        self.scale[variable] = self.scale[variable] * factor

    def split(self, frac):
        """Method to split dataset into training and validation subsets

        Args:
            frac (float): the fraction to split the dataset into the training set. The training set will be indexed from `0` to `frac` $\\times$ `Dataset.n`. The validation dataset will be from the last index of the training set to the last row of the dataset.

        Note:
            The actual splitting of the dataset is done during the training procedure,
            or when invoking the `train_dataset()` or `valid_dataset()` methods

        """

        self.split_frac = frac
        info(f"n_train_samples:{self.n_train} n_valid_samples:{self.n_valid}")

    def _dataset_slice(self, tensors, index, batch_size, shift, n_index):
        """Internal method call for self.train_dataset or self.valid_dataset"""

        if not isinstance(tensors, list):
            tensors = [tensors]

        # check if all tensors are of the same type tensors
        if all(isinstance(t, TensorVariable) for t in tensors):
            pass
        # if tensor inputs are list of strings, convert them to tensors
        elif all(isinstance(t, str) for t in tensors):
            tensors = [self[t] for t in tensors]
        else:
            raise TypeError(f"Multiple types found in {tensors}.")

        # retrieve tensor names
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
        """Returns a slice of the (or the full) training data array with the sequence
        matching the list of variables.

        Args:
            variables (Union[list, str, TensorVariable]): a tensor, label, or list of
                tensors or list of labels
            index (int): the start of the slice of the data array. If `None` is given,
                returns the full data array.
            batch_size (int): length of the slice. If `None` is given, returns the
                index from `index` to `N` where `N` is the length of the array.
            shift (int): the offset of the slice between `0` and `batch_size`. If
                `None` is given, `shift=0`.

        Returns:
            (list): a list of array object(s) corresponding to the input variables

        !!! Example
            How to retrieve data array from Dataset:
            ```python
            ds = Dataset(df, choice="choice")

            # index "age" and "location" data arrays
            return ds.train_dataset([ds["age"], ds["location"]])

            # similar result
            return ds.train_dataset(["age", "location"])
            ```
        """

        n_index = self.train_index

        return self._dataset_slice(variables, index, batch_size, shift, n_index)

    def valid_dataset(self, variables, index=None, batch_size=None, shift=None):
        """Returns a slice of the (or the full) validation data array with the sequence
        matching the list of variables.

        Args:
            variables (Union[list, str, TensorVariable]): a tensor, label, or list of
                tensors or list of labels
            index (int): the start of the slice of the data array. If `None` is given,
                returns the full data array.
            batch_size (int): length of the slice. If `None` is given, returns the
                index from `index` to `N` where `N` is the length of the array.
            shift (int): the offset of the slice between `0` and `batch_size`. If
                `None` is given, `shift=0`.

        Returns:
            (list): a list of array object(s) corresponding to the input variables
        """

        n_index = self.valid_index

        return self._dataset_slice(variables, index, batch_size, shift, n_index)
