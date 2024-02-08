"""
The code snippet is a part of a class called `Dataset` that converts a pandas DataFrame into an xarray dataset. It initializes the dataset object with the DataFrame and the name of the choice variable. It also provides methods to access and manipulate the dataset.
"""

import aesara.tensor as aet
from aesara.tensor.var import TensorVariable

import pycmtensor.defaultconfig as defaultconfig

config = defaultconfig.config

from pycmtensor.logger import debug, info

__all__ = ["Dataset"]


class Dataset:
    def __init__(self, df, choice, shuffle=True, **kwargs):
        """Initialize the Dataset object with a pandas DataFrame and the name of the choice variable.

        Args:
            df (pandas.DataFrame): The pandas DataFrame object containing the dataset.
            choice (str): The name of the choice variable.
            shuffle (bool): Whether to shuffle the dataset. Default is True.
            **kwargs (optional): Additional keyword arguments to configure the dataset.

        Attributes:
            n (int): The number of rows in the dataset.
            x (list[TensorVariable]): The list of input TensorVariable objects.
            y (TensorVariable): The output TensorVariable object.
            scale (dict): A dictionary of scaling factors for each variable.
            choice (str): The name of the choice variable.
            ds (dict): A dictionary of variable values.
            split_frac (float): The split fraction used to split the dataset.
            idx_train (list): The list of indices of the training dataset.
            idx_valid (list): The list of indices of the validation dataset.
            n_train (int): The size of the training dataset.
            n_valid (int): The size of the validation dataset.

        Example:
            Example initialization of a Dataset object:

            ```python
            ds = Dataset(df=pd.read_csv("datafile.csv", sep=","), choice="mode")
            ds.split(frac=0.8)
            ```

            Accessing attributes:
            ```python
            print(ds.choice)
            ```

            Output:
            ```bash
            'car'
            ```

        Raises:
            IndexError: If the choice variable is not found in the DataFrame columns.
        """
        # Iterate over the keyword arguments and add them to the config
        for key, value in kwargs.items():
            config.add(key, value)

        # Check if the choice is in the dataframe columns
        if choice not in df.columns:
            raise IndexError(f"{choice} not found in dataframe.")

        # Convert the choice column to integer type
        df[choice] = df[choice].astype("int")

        # Reset the dataframe index
        df.reset_index(drop=True, inplace=True)

        # Normalize the choice column to start from 0
        while df[choice].min() > 0:
            df[choice] -= df[choice].min()

        # Store the dataframe index values
        self.index = df.index.values

        # Initialize the variables
        self.n = len(df)
        self.x = []
        self.y = None

        # Iterate over the dataframe columns
        for name in list(df.columns):
            if name == choice:
                self.y = aet.ivector(name)
            else:
                self.x.append(aet.vector(name))

        # Shuffle the dataframe
        if shuffle:
            df = df.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)

        # Create a dictionary with column names as keys and column values as values
        ds = {name: df[name].values for name in list(df.columns)}

        # Initialize the scale with variable names and 1.0 as values
        self.scale = {var.name: 1.0 for var in self.x}

        # Store the choice and the dictionary
        self.choice = choice
        self.ds = ds

        # Set the split fraction
        self.split_frac = 1.0

    def __call__(self):
        return self.ds

    def __getitem__(self, key):
        """Returns the input or output variable(s) of the dataset object by their names.

        Args:
            key (str or list or tuple): The name(s) of the variable(s) to be accessed.

        Returns:
            TensorVariable or list of TensorVariable: The input or output variable(s) corresponding to the given name(s).

        Raises:
            KeyError: If the given name(s) do not match any input or output variable.
        """
        # Check if the key is a list or tuple
        if isinstance(key, (list, tuple)):
            # If it is, convert it to a tensor
            return self._make_tensor(key)
        else:
            # If the key is a name of one of the x variables
            if key in [var.name for var in self.x]:
                # Return the corresponding x variable
                return self.x[[x.name for x in self.x].index(key)]

            # If the key is the name of the y variable
            if key == self.y.name:
                return self.y

            # If the key is not found in the x or y variables
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
        return len(self.idx_train)

    @property
    def n_valid(self) -> int:
        return len(self.idx_valid)

    @property
    def idx_train(self) -> list:
        if self.split_frac == 1:
            return self.index
        n = round(self.n * self.split_frac)
        return self.index[:n]

    @property
    def idx_valid(self) -> list:
        if self.split_frac == 1:
            return self.index
        n = round(self.n * self.split_frac)
        return self.index[n:]

    def drop(self, variables) -> None:
        """Method for dropping `variables` from the dataset

        Args:
            variables (list[str]): list of `str` variables from the dataset to drop

        Raises:
            KeyError: raises an error if any item in `variables` is not found in the dataset or item is the choice variable

        !!! Warning
            Choice variable cannot be explicitly dropped.
        """
        for variable in variables:
            if variable == self.choice:
                raise KeyError(f"Cannot drop choice variable '{variable}'")

            if variable in self.ds:
                i = [x.name for x in self.x].index(variable)
                del self.x[i]
                del self.scale[variable]
                del self.ds[variable]
                debug(f"Dropped input variable '{variable}' from dataset")

            else:
                raise KeyError(f"Variable '{variable}' not found in dataset")

    def scale_variable(self, variable, factor) -> None:
        """Multiply values of the `variable` by `1/factor`.

        Args:
            variable (str or list[str]): the name of the variable or a list of variable names
            factor (float): the scaling factor
        """
        if not isinstance(variable, list):
            variable = [variable]

        for var in variable:
            if not isinstance(var, str):
                raise TypeError(f"Variable '{var}' must be a string")

            self.ds[var] = self.ds[var] / factor
            self.scale[var] = self.scale[var] * factor

    def split(self, frac=None, count=None) -> None:
        """Method to split the dataset into training and validation subsets based on a given fraction.

        Args:
            frac (float): The fraction to split the dataset into the training set.
            count (int): The number of samples to split the dataset into the training set.

        Returns:
            None

        Notes:
            - The actual splitting of the dataset is done during the training procedure or when invoking the `train_dataset()` or `valid_dataset()` methods.
            - Either frac or count must be specified. If both are specified, count is ignored.
        """
        if frac is None and count is None:
            raise ValueError("Either frac or count must be specified")
        if frac is not None and count is not None:
            debug("count is ignored")
        if frac is not None:
            self.split_frac = frac
        else:
            self.split_frac = count / self.n
            assert self.split_frac <= 1.0

        info(
            f"seed: {config.seed} n_train_samples:{self.n_train} n_valid_samples:{self.n_valid}"
        )

    def _dataset_slice(self, tensors, index, batch_size, shift, n_index) -> list:
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

    def train_dataset(self, variables, index=None, batch_size=None, shift=None) -> list:
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

        n_index = self.idx_train

        return self._dataset_slice(variables, index, batch_size, shift, n_index)

    def valid_dataset(self, variables, index=None, batch_size=None, shift=None) -> list:
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

        n_index = self.idx_valid

        return self._dataset_slice(variables, index, batch_size, shift, n_index)
