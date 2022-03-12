import pytest
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb

from pycmtensor.utils import tqdm_nb_check


def test_nb_check():
    notebook = True
    pbar = tqdm_nb_check(notebook)
    assert pbar == tqdm_nb

    notebook = False
    pbar = tqdm_nb_check(notebook)
    assert pbar == tqdm


def test_nb_check_exception():
    with pytest.raises(Exception) as excinfo:
        tqdm_nb_check(123)
    assert str(excinfo.value) == "123 is not a bool type"
