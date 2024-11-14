Experimenting
=============

Running common tasks with :code:`nox`
-------------------------------------

The fastest way to start with development is to use nox. If you don't have nox, you can use :code:`pipx run nox`
to run it without installing, or :code:`pipx install nox`. If you don't have pipx (pip for applications), then
you can install with :code:`pip install pipx` (the only case were installing an application with regular
pip is reasonable). If you use macOS, then pipx and nox are both in brew, use :code:`brew install pipx nox`.

To use, run :code:`nox`. This will lint and test using every installed version of Python on your system,
skipping ones that are not installed. You can also run specific jobs:

.. code-block:: bash

    python -m nox -s lint  # Lint only
    python -m nox -s tests  # Python tests
    python -m nox -s docs -- --serve  # Build and serve the docs
    python -m nox -s build  # Make an SDist and wheel
    python -m nox -s jupyter  # Run jupyter notebooks

Nox handles everything for you, including setting up an temporary virtual environment for each run.

Jupyter
-------

You can run Jupyter notebooks using:

.. code-block:: bash

    python -m nox -s jupyter

.. note::
    You need to have `SCIP <https://www.scipopt.org/>`_ installed. See :ref:`installation` for details.

.. toctree::
    :maxdepth: 2

    installation
    nox
    notebooks
    experiments
    plgrid
    database
