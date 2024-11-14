Setting up
==========

Setting up a research project
-----------------------------



Setting up a development environment
------------------------------------

In a minimal installation, you should be able to run :code:`stackelberg-games-core` unit tests
and build the documentation.

.. code-block:: bash

    python -m venv .venv --prompt sg
    source .venv/bin/activate
    python -m pip install -r requirements-docs.txt

    python -m pip install -r requirements-dev.txt
    python -m pip install --no-build-isolation -v -e stackelberg-games-core/

Pre-commit
----------

You should prepare pre-commit, which will help you by checking that commits pass required checks:

.. code-block:: bash

    pip install pre-commit # or brew install pre-commit on macOS
    pre-commit install # Will install a pre-commit hook into the git repo


You can also/alternatively run :code:`pre-commit run` (changes only) or :code:`pre-commit run --all-files`
to check even without installing the hook.

Github Actions
--------------
