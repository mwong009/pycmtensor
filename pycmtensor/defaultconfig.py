# defaultconfig.py
"""PyCMTensor config module"""
import configparser
import multiprocessing
import os

__all__ = ["Config"]


class Config:
    """Config class object that holds configuration settings

    Attributes:
        descriptions (dict): descriptive documentation of each configuration setting

    !!! tip
        To display a current list of configuration settings, invoke `print(pycmtensor.config)`.

        ```python
        import pycmtensor
        print(pycmtensor.config)
        ```

        Output:
        ```bash
        PyCMTensor configuration
        ...
        ```
    """

    def __init__(self):
        self.descriptions = {}
        self.ACCEPT_LOGLIKE = [
            1,
            "loglike",
            "loglikelihood",
            "likelihood",
            "max_loglikelihood",
            "max loglikelihood",
        ]
        self.TIME_COUNTER = 0

    def __repr__(self):
        msg = "PyCMTensor configuration\n"
        for key, val in self.__dict__.items():
            if key != "descriptions":
                msg += f"{key}\n"
                msg += f"    Value:  {val} {type(val)}\n"

                if key in self.descriptions:
                    msg += f"    Doc:    {self.descriptions[key]}\n"

                msg += f"\n"

        return msg

    def update(self, name, value, *args, **kwargs):
        """update the config parameter

        Args:
            name (str): the name of the parameter
            value (any): the value of the parameter
            *args (None): overloaded arguments
            **kwargs (dict): overloaded keyword arguments
        """
        if name in dir(self):
            if type(value) == type(getattr(self, name)):
                setattr(self, name, value)
            else:
                ftype = str(type(getattr(self, name))).split("'")[1]
                raise TypeError(f"Incorrect type for {name}. ({ftype})")
        else:
            raise KeyError(f"{name} not a valid config parameter")

    def add(self, name, value, description=None):
        """Method to add a new or update a setting in the configuration

        Args:
            name (str): the name of the parameter
            value (any): the value of the parameter
            description (str): description of the parameter

        !!! example
            To set the value of the random seed to 100
            ```python
            pycmtensor.config.add('seed', 100, description='seed value')
            ```
        """
        if name in dir(self):
            self.update(name, value)
        else:
            setattr(self, name, value)

            if description is not None:
                self.descriptions[name] = description


config = Config()

# check number of cores on system for multiprocessing

num_cores = multiprocessing.cpu_count()
os.environ["MKL_NUM_THREADS"] = str(num_cores)
os.environ["OMP_NUM_THREADS"] = str(num_cores)

# write .pytensorrc file

HOMEPATH = os.path.expanduser("~")
conf_file = os.path.join(HOMEPATH, ".pytensorrc")
pytensor_rc = configparser.ConfigParser()
pytensor_rc.add_section("global")
pytensor_rc["global"] = {"device": "cpu", "floatX": "float64"}

with open(conf_file, "w") as f:
    pytensor_rc.write(f)
