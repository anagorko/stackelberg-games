:sd_hide_title:

Stackelberg Games Repository
============================

.. div::
    :style: padding: 0.1rem 0.5rem 0.6rem 0; background-image: linear-gradient(315deg, #11111f 0%, #341a61 74%); clip-path: polygon(0px 0px, 100% 0%, 100% 100%, 0% calc(100% - 1.5rem)); -webkit-clip-path: polygon(0px 0px, 100% 0%, 100% 100%, 0% calc(100% - 1.5rem));

    .. grid::
        :gutter: 2 3 3 3
        :margin: 4 4 1 2

        .. grid-item::
            :columns: 12 9 9 9
            :child-align: justify
            :class: sd-text-white sd-fs-4

            A repository with research
            projects by AI for Security research group.

        .. grid-item::
            :columns: 6 1 1 1

            .. button-ref:: publications
                :ref-type: ref
                :outline:
                :color: white
                :class: sd-px-4 sd-fs-6

                Cite

        .. grid-item::
            :columns: 6 1 1 1

            .. button-link:: https://github.com/anagorko/stackelberg_games
                :ref-type: ref
                :outline:
                :color: white
                :class: sd-px-4 sd-fs-6

                GitHub



.. grid:: 1 2 2 2
    :gutter: 1
    :margin: 4 0 0 0

    .. grid-item-card:: Stackelberg Games Core :octicon:`link-external;1em`
        :link: stackelberg-games-core/index
        :link-type: doc

        A repository with reference implementations of algorithms and data structures to serve
        as a baseline in Bayesian Stackelberg research.

        Apart from both classical and novel algorithms, the repository provides visualization tools and
        a support for cluster computing.

    .. grid-item-card:: Experiment Replication :octicon:`link-external;1em`
        :link: projects
        :link-type: doc

        A collection of archival repositories with source code to replicate experiments as published in our papers.

        Once published, these algorithms are either ported to Stackelberg Games Core or moved
        to new research projects for further development.

    .. grid-item-card:: Whitepaper :octicon:`link-external;1em`
        :link: whitepaper/whitepaper
        :link-type: doc

        A :doc:`set of notes <whitepaper/whitepaper>` with theoretical background for Bayesian Stackelberg games research.

    .. grid-item-card:: Contributing Guidelines :octicon:`link-external;1em`
        :link: develop/contributing
        :link-type: doc

        Contributions to Benchmark, Core and Whitepaper are welcome.
        Read :doc:`Contributing Guidelines <develop/contributing>` and submit a pull request.

        Our active research projects are kept private to not disrupt the review process.

.. note::

    Project is under heavy development. Please report outdated docs to the project maintainer, Andrzej Nagórko.

.. _publications:

Publications
------------

.. note::

    You may cite the entire repository as

    :bdg-warning:`to be decided`

    You may also want to cite individual publications, as listed below.

.. bibliography::
    :style: unsrt
    :filter: author % "Andrzej"

Repositories
------------

`stackelberg-games <https://github.com/anagorko/stackelberg_games>`_ is a monorepo with general-purpose
repositories :code:`stackelberg-games-core/` and :code:`stackelberg-games-benchmark/`, a set of notes
about Bayesian Stackelberg games as well as
archival repositories with code to replicate experiments from our papers.

.. note::
    See :doc:`experiment/guide` for general instructions (e.g. cluster computing) and individual
    :code:`README.md` files for instructions specific to a project.

.. grid:: 1 2 2 2
    :gutter: 1
    :margin: 4 0 0 0

    .. grid-item-card::

        Documentation
        ^^^

        Source code: `docs/ <https://github.com/anagorko/monte-carlo-methods/tree/main/docs>`_.

        +++
        Contributors: Andrzej Nagórko, Marcin Waniek.

    .. grid-item-card::

        Whitepaper
        ^^^

        Source code: `docs/whitepaper/ <https://github.com/anagorko/monte-carlo-methods/tree/main/docs/whitepaper>`_.

        +++
        Contributors: Andrzej Nagórko.

    .. grid-item-card::

        Stackelberg Games Core
        ^^^

        Source code: `stackelberg-games-core/ <https://github.com/anagorko/monte-carlo-methods/tree/main/stackelberg-games-core>`_.

        +++
        Contributors: Andrzej Nagórko, Marcin Waniek, Łukasz Gołuchowski.

    .. grid-item-card::

        Two-phase Games
        ^^^

        Source code: `stackelberg-games-twophase/ <https://github.com/anagorko/monte-carlo-methods/tree/main/stackelberg-games-twophase>`_.

        +++
        Contributors: Andrzej Nagórko, Paweł Ciosmak, Tomasz Michalak

    .. grid-item-card::

        Shield algorithm
        ^^^

        +++
        Contributors: Andrzej Nagórko, Michał Tomasz Godziszewski, Marcin Waniek, Barbara Rosiak, Małgorzata Róg, Tomasz Michalak.

License
-------

Source code published in this repository
is licensed under :doc:`Apache License Version 2.0 <license>`, with the following exceptions.

* :code:`stackelberg-games/core/algorithms/dots.py` is ported from ... and is licensed under MIT license

.. toctree::
    :maxdepth: 3
    :hidden:

    Core <stackelberg-games-core/index.rst>

.. toctree::
    :maxdepth: 3
    :hidden:

    Experimenting <experiment/index.rst>

.. toctree::
    :maxdepth: 3
    :hidden:

    Whitepaper <whitepaper/index.rst>

.. toctree::
    :maxdepth: 3
    :hidden:

    Contributing <contributing/index.rst>
