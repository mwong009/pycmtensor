# database.py

import aesara
import aesara.tensor as aet
import biogeme.database as biodb
import numpy as np

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
            raise NotImplementedError(f"Variable {item} not found")

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

    def get_x_tensors(self):
        x_tensors = []
        for _, variable in self.variables.items():
            if hasattr(variable, "x"):
                x_tensors.append(variable.x)
        return x_tensors

    def get_x_data(self, index=None, batch_size=None, shift=None):
        x_data = []
        x_tensors = self.get_x_tensors()
        for x_tensor in x_tensors:
            if index == None:
                x_data.append(self.data[x_tensor.name])
            else:
                start = index * batch_size + shift
                end = (index + 1) * batch_size + shift
                x_data.append(self.data[x_tensor.name][start:end])
        return x_data

    def get_y_tensors(self):
        y_tensors = []
        for _, variable in self.variables.items():
            if hasattr(variable, "y"):
                y_tensors.append(variable.y)
        return y_tensors

    def get_y_data(self, index=None, batch_size=None, shift=None):
        y_data = []
        y_tensors = self.get_y_tensors()
        for y_tensor in y_tensors:
            if index == None:
                y_data.append(self.data[y_tensor.name])
            else:
                start = index * batch_size + shift
                end = (index + 1) * batch_size + shift
                y_data.append(self.data[y_tensor.name][start:end])
        return y_data

    def tensors(self):
        return self.get_x_tensors() + self.get_y_tensors()

    def input_data(self, index=None, batch_size=None, shift=None):
        """Outputs a list of pandas table data corresponding to the
        Symbolic variables

        Args:
            index (int, optional): Starting index slice.
            batch_size (int, optional): Size of slice.
            shift (int, optional): Add a random shift of the index.

        Returns:
            list: list of database (input) data
        """
        return self.get_x_data(index, batch_size, shift) + self.get_y_data(
            index, batch_size, shift
        )

    def input_shared_data(self):
        """Outputs a list of sharedVariable data corresponding to the symbolic
        variables

        Returns:
            list: a list  of sharedVariable database (input) data
        """
        shared_data = []
        tensors = self.get_x_tensors() + self.get_y_tensors()
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
                    print("scaling {} by {}".format(d, scale))
