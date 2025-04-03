from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import nox

DIR = Path(__file__).parent.resolve()

nox.options.sessions = ["lint", "pylint", "tests"]


@nox.session
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests.
    """
    session.install("-C", "build-dir=", "-C", "editable.rebuild=false", ".[test]")
    session.run("pytest", *session.posargs)


@nox.session(reuse_venv=True)
def docs(session: nox.Session) -> None:
    """
    Build the docs. Pass "--serve" to serve. Pass "-b linkcheck" to check links.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Serve after building")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    parser.add_argument(
        "-b", dest="builder", default="html", help="Build target (default: html)"
    )
    args, posargs = parser.parse_known_args(session.posargs)

    if args.builder != "html" and args.serve:
        session.error("Must not specify non-HTML builder with --serve")

    extra_installs = ["sphinx-autobuild"] if args.serve else []

    session.install("-r", "requirements-docs.txt")
    session.install("-e", "stackelberg-games-core/")
    session.install("-e", "stackelberg-games-patrolling/")
    if extra_installs:
        session.install(*extra_installs)
    session.chdir("docs")

    if args.builder == "linkcheck":
        session.run(
            "sphinx-build", "-b", "linkcheck", ".", "_build/linkcheck", *posargs
        )
        return

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        ".",
        f"_build/{args.builder}/c40b6000c0a0ef9057debc8c727d33ae37bc5127",
        *posargs,
    )

    if args.serve:
        session.run("sphinx-autobuild", f"--port={args.port}", *shared_args)
    else:
        session.run("sphinx-build", "--keep-going", *shared_args)


@nox.session(reuse_venv=True)
def jupyter(session: nox.Session) -> None:
    """
    Run jupyter-lab server.
    """

    session.install(
        "-C", "build-dir=", "-C", "editable.rebuild=false", "stackelberg-games-core/"
    )
    session.install("jupyterlab", "ipykernel")
    session.run("python", "-m", "ipykernel", "install", "--user", "--name", "sgmc-nox")
    session.chdir("notebooks")

    session.run("jupyter-lab")
