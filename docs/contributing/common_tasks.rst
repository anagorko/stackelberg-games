Common Tasks
============

Building docs
-------------

You cannot publish docs from a private repository to :code:`readthedocs.io`.
If you are working in a private fork, you can view the documentation locally using
the following command.

.. code-block:: bash

    sphinx-autobuild -n -T -b=html docs/ docs/_build/html/

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
