Setting up
==========

Setting up a research project
-----------------------------

We set up new research projects in private repositories to not interfere with blind review.
As Github does not allow private forks of public repositories, we create the copy manually.

1. Create a new private repository on Github, e.g. :code:`stackelberg-games-twophase`
and set the environment variable to the repository URL.

.. code-block:: bash

    CLONE_URL=git@github.com:anagorko/stackelberg-games-twophase.git

2. Chdir to the :code:`stackelberg-games` repository and run and push its contents to the new
repository.

.. code-block:: bash

    cd stackelberg-games/
    git push $CLONE_URL
    cd ..

3. Clone the copy and set upstream to :code:`stackelberg-games`.

.. code-block:: bash

    git clone ${CLONE_URL}
    PREFIX=${CLONE_URL%.*}
    cd ${PREFIX##*/}
    git remote add upstream git@github.com:anagorko/stackelberg-games.git
    git remote set-url --push upstream DISABLE
    git remote -v

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
