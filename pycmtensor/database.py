# database.py

import logging

import aesara
import aesara.tensor as aet
import biogeme.database as biodb
import numpy as np

from pycmtensor import logger as log
from pycmtensor.logger import PyCMTensorError

floatX = aesara.config.floatX


class Database(biodb.Database):
    def __init__(self, name, pandasDatabase, choiceVar):
        super().__init__(name, pandasDatabase)
        assert choiceVar in self.data.columns
        for name, variable in self.variables.items():
            if name in self.data.columns:
                if name == choiceVar:
                    variable.y = aet.ivector(name)
                else:
                    variable.x = aet.vector(name)
        self.choiceVar = self[choiceVar]
        log.info(f"Choice variable set as '{self.choiceVar}'")

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getattr__(self, name):
        if name in self.variables:
            return self.variables[name]

    def __getitem__(self, item):
        """Returns the aesara.tensor.var.TensorVariable object.
        Use `Database["columnName"]` to reference the TensorVariable
        """
        if hasattr(self.variables[item], "x"):
            return self.variables[item].x
        elif hasattr(self.variables[item], "y"):
            return self.variables[item].y
        else:
            msg = f"Variable {item} not found"
            log.error(msg)
            raise PyCMTensorError(msg)

    def compile_data(self):
        self.sharedData = {}
        for name, variable in self.variables.items():
            if name in self.data.columns:
                shared_data = aesara.shared(
                    np.asarray(self.data[name], dtype=floatX), borrow=True, name=name
                )
                if hasattr(variable, "y"):
                    self.sharedData[name] = aet.cast(shared_data, "int32")
                else:
                    self.sharedData[name] = shared_data

    def get_rows(self):
        """Get the number of observations (row) in the database

        Returns:
            int: The number of rows in the dataset
        """
        return len(self.data)

    def get_x_tensors(self):
        x_tensors = []
        for _, variable in self.variables.items():
            if hasattr(variable, "x"):
                x_tensors.append(variable.x)
        return x_tensors

    def get_y_tensors(self):
        y_tensors = []
        for _, variable in self.variables.items():
            if hasattr(variable, "y"):
                y_tensors.append(variable.y)
        return y_tensors

    def get_tensors(self):
        """Returns all the tensors (x, y) in the database

        Returns:
            list: ``x`` and ``y`` tensors are returned as a list
        """
        return self.get_x_tensors() + self.get_y_tensors()

    def input_data(self, inputs=None, index=None, batch_size=None, shift=None):
        """Outputs a list of pandas table data corresponding to the
        Symbolic variables

        Args:
            inputs (list, optional): if inputs are given, returns the sharedVariable that exists in inputs. If None, return all sharedVariable in this database.
            index (int, optional): Starting index slice.
            batch_size (int, optional): Size of slice.
            shift (int, optional): Add a random shift of the index.

        Returns:
            list: list of pandas array data
        """
        data = []
        if inputs is not None:
            tensors = inputs
        else:
            tensors = self.get_tensors()
        for tensor in tensors:
            if index == None:
                data.append(self.data[tensor.name])
            else:
                start = index * batch_size + shift
                end = (index + 1) * batch_size + shift
                data.append(self.data[tensor.name][start:end])

        return data

    def input_shared_data(self, inputs: list = None):
        """Outputs a list of sharedVariable data corresponding to the symbolic
        variables

        Args:
            inputs (list, optional): if inputs are given, returns the sharedVariable that exists in inputs. If None, return all sharedVariable in this database.

        Returns:
            list: a list of sharedVariable tensor data
        """
        shared_data = []
        if inputs is not None:
            tensors = inputs
        else:
            tensors = self.get_tensors()
        for tensor in tensors:
            shared_data.append(self.sharedData[tensor.name])
        return shared_data

    def autoscale(self, variables=None, verbose=False, default=None):
        for d in self.data:
            max_val = np.max(self.data[d])
            min_val = np.min(self.data[d])
            scale = 1.0
            if variables is None:
                varlist = self.get_x_tensors()
                varlist = [v.name for v in varlist]
            else:
                varlist = variables
            if d in varlist:
                if min_val >= 0.0:
                    if default is None:
                        while max_val > 10:
                            self.data[d] /= 10.0
                            scale *= 10.0
                            max_val = np.max(self.data[d])
                    else:
                        self.data[d] /= default
                        scale = default
                if verbose:
                    log.info(f"scaling {d} by {scale}")
