.. highlight:: shell

============
Installation
============

Requirements
------------

.. note ::
    
    We only support installation of requirements through `Conda`_ and `pip`_.

.. margin ::

    Test margin

.. list-table::
    :widths: 20 25 55
    :header-rows: 1

    * - Module
      - Version
      - 
    * - Python_
      - == 3.9.7
      - Other versions of python (>3.6) might be supported. Python version 3.10 and above are not yet supported.
    * - Numpy_
      - >= 1.21.0
      - Older versions of numpy may be compatible.
    * - Scipy_
      - >= 1.7.1, <1.8.0
      - Scipy 1.8.0 and above may have bugs.
    * - `Intel MKL`_
      - >= 2022.0.2
      - Intel® Math Kernel Library (installed with `mkl-service`) provides the BLAS libraries. It is recommended that `mkl-service` is installed through conda
    * - Biogeme_
      - >= 3.2.8
      - Latest version of biogeme can be downloaded from `<https://github.com/michelbierlaire/biogeme>`_
    * - Aesara_
      - >= 2.4.0
      - Latest version of Aesara can be downloaded from `<https://github.com/aesara-devs/aesara>`_

.. _Python: https://www.python.org/
.. _Numpy: https://numpy.org/
.. _Scipy: https://scipy.org/
.. _Intel MKL: https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-mkl-for-dpcpp/top.html
.. _Biogeme: https://biogeme.epfl.ch/
.. _Aesara: https://aesara.readthedocs.io/en/latest/index.html

Installing through Conda (miniconda)
------------------------------------

Download and install `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.

Run the following command to install the required packages:

.. code-block:: console

    $ conda install -c conda-forge git cxx-compiler m2w64-toolchain libblas libpython

Stable release
--------------

To install `PyCMTensor`, run this command in your terminal:

.. code-block:: console

    $ pip install pycmtensor

This is the preferred method to install `PyCMTensor`, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

We recommend using `Conda`_ as the dependency and package manager. 

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/
.. _Conda: https://https://docs.conda.io/en/latest/miniconda.html.pypa.io

From sources
------------

The sources for PyCMTensor can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/mwong009/pycmtensor

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/mwong009/pycmtensor/tarball/master

Once you have a copy of the source, `cd` to the repo directory and install with:

.. code-block:: console

    $ cd pycmtensor
    $ make install

.. _Github repo: https://github.com/mwong009/pycmtensor
.. _tarball: https://github.com/mwong009/pycmtensor/tarball/master
