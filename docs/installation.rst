.. highlight:: shell

============
Installation
============

Requirements
------------

.. note ::
    
    We only support installation of requirements through `Conda`_.

.. margin ::

    Test margin

.. list-table::
    :widths: 20 25 55
    :header-rows: 1

    * - Module
      - Version
      - 
    * - Python_
      - >= 3.9
      - Other versions of python (>3.6) might be supported. Python version 3.10 and above are now supported.
    * - Numpy_
      - >= 1.21.0
      - Older versions of numpy may be compatible.
    * - Scipy_
      - >= 1.8.0
      - 
    * - `Intel MKL`_
      - >= 2022.0.1 
      - Intel® Math Kernel Library (installed with `mkl-service`) provides the BLAS libraries. It is recommended that `mkl-service` is installed through conda
    * - Aesara_
      - >= 2.4.0
      - Latest version of Aesara can be downloaded from `<https://github.com/aesara-devs/aesara>`_

.. _Python: https://www.python.org/
.. _Numpy: https://numpy.org/
.. _Scipy: https://scipy.org/
.. _Intel MKL: https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-mkl-for-dpcpp/top.html
.. _Aesara: https://aesara.readthedocs.io/en/latest/index.html

Installing through Conda (miniconda)
------------------------------------

Download and install `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.

Run the following command to install the required packages:

To install PyCMTensor, you need `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ (Full Anaconda works fine, but **miniconda** is recommmended for a minimal installation). Ensure that Conda is using at least `Python 3.9`.

Once Conda is installed, install the required dependencies from conda by running the following command in your terminal:

Windows:

.. code-block:: console

    conda install mkl-service conda-forge::cxx-compiler conda-forge::m2w64-toolchain -y

Mac OS X:

.. code-block:: console

    conda install mkl-service Clang

Linux:

.. code-block:: console

    conda install mkl-service conda-forge::cxx-compiler


Stable release
--------------

Run this command in your terminal to download and install the latest branch of `PyCMTensor` from `PyPI <https://pypi.org/project/pycmtensor/>`_:

.. code-block:: console

    pip install pycmtensor -U

This is the preferred method to install `PyCMTensor`, as it will always install the most recent stable release.

*Optional*: If you want the development version from the Github repository:

.. code-block:: console

    pip install git+https://github.com/mwong009/pycmtensor.git@develop -U

We recommend using `Conda`_ as the dependency and package manager. 

.. _Conda: https://https://docs.conda.io/en/latest/miniconda.html.pypa.io


Development
-----------

The sources for PyCMTensor can be downloaded from the `Github repo`_.

.. code-block:: console

    git clone git://github.com/mwong009/pycmtensor

To set up `PyCMTensor` in a local development environment, you need to set up a virtual environment and install the project requirements. Follow the instructions to install Conda (miniconda), then start a new virtual environment with the provided `environment_<your OS>.yml` file.

For example in windows:

.. code-block:: console

    conda env create -f environment_windows.yml

Next, activate the virtual environment and install poetry via `pip`.

.. code-block:: console

    conda activate pycmtensor-dev
    pip install poetry

Lastly, install the project and development dependencies

.. code-block:: console

    poetry install -E dev

.. _Github repo: https://github.com/mwong009/pycmtensor