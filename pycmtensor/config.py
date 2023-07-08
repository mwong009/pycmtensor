# config.py
"""PyCMTensor config module"""
import configparser
import multiprocessing
import os

__all__ = ["Config"]


class Config:
    def __init__(self):
        self.descriptions = {}

    def __repr__(self):
        msg = "Model parameters\n"
        for key, val in self.__dict__.items():
            if key != "descriptions":
                msg += f"{key}\n"
                msg += f"    Value:  {val} {type(val)}\n"

                if key in self.descriptions:
                    msg += f"    Doc:    {self.descriptions[key]}\n"

                msg += f"\n"

        return msg

    def add(self, name, value, description=None):
        """Add a new parameter to Config"""

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
