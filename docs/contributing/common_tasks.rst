Common Tasks
============

Building docs
-------------

You cannot publish docs from a private repository to `readthedocs.io`_.

You can build the docs using:

.. code-block:: bash

    python -m nox -s docs

You can see a preview with:

.. code-block:: bash

    python -m nox -s docs -- --serve

Jupyter
-------

.. code-block:: bash

    cd notebooks/
    jupyter-lab

Testing
-------

Use pytest to run the unit checks:

.. code-block:: bash

    python -m pytest

Coverage
--------

Use pytest-cov to generate coverage reports:

.. code-block:: bash

    python -m pytest --cov=stackelberg-games.core
