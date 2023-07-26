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

    def add(self, name, value: any, description=None):
        """Method to add a new or update a setting in the configuration

        Args:
            name (str): name of the setting
            value: value given to the setting
            description (str): a string text describing the function of the setting

        !!! example
            To set the value of the random seed to 100
            ```python
            pycmtensor.config.add('seed', 100)
            ```
        """

        if name in self.__dict__:
            # check same instance as existing parameter
            if not isinstance(value, type(getattr(self, name))):
                raise TypeError(f"{name} must of of type {type(getattr(self, name))}.")

        setattr(self, name, value)

        if description is not None:
            self.descriptions[name] = description


config = Config()

# check number of cores on system for multiprocessing

num_cores = multiprocessing.cpu_count()
os.environ["MKL_NUM_THREADS"] = str(num_cores)
os.environ["OMP_NUM_THREADS"] = str(num_cores)

# write .aesararc file

HOMEPATH = os.path.expanduser("~")
conf_file = os.path.join(HOMEPATH, ".aesararc")
aesara_rc = configparser.ConfigParser()
aesara_rc.add_section("global")
aesara_rc["global"] = {"device": "cpu", "floatX": "float64"}

with open(conf_file, "w") as f:
    aesara_rc.write(f)
