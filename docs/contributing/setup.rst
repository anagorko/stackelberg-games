Setting up
==========

Setting up a research project
-----------------------------

We set up new research projects in private repositories to not interfere with blind review.
As Github does not allow private forks of public repositories, we duplicate :code:`stackelberg-games`
following `these steps <https://gist.github.com/0xjac/85097472043b697ab57ba1b1c7530274>`_.

1. Create a new private repository on Github, e.g. :code:`sg-tutorial` and set the environment variable
with the repository URL.

.. code-block:: bash

    CLONE_URL=git@github.com:anagorko/sg-tutorial.git

2. Run the following commands.

.. code-block:: bash

    CLONE_URL_STRIPPED=${CLONE_URL%.*}
    CLONE_DIR=${CLONE_URL_STRIPPED##*/}

    mkdir tmp
    cd tmp
    git clone --bare git@github.com:anagorko/stackelberg-games.git
    cd stackelberg-games.git/
    git push --mirror $CLONE_URL
    cd ..
    rm -rf stackelberg-games.git/
    cd ..
    rmdir tmp
    git clone $CLONE_URL
    cd $CLONE_DIR
    git remote add upstream git@github.com:anagorko/stackelberg-games.git
    git remote set-url --push upstream DISABLE
    git remote -v

The output should be

.. code-block:: terminal

    origin	git@github.com:anagorko/sg-tutorial.git (fetch)
    origin	git@github.com:anagorko/sg-tutorial.git (push)
    upstream	git@github.com:anagorko/stackelberg-games.git (fetch)
    upstream	DISABLE (push)

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
