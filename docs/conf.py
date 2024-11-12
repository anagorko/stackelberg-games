from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

sys.path.insert(0, Path("../src").resolve().name)
sys.path.append(str(Path(".").resolve()))


project = "stackelberg-games"
copyright = "2024, Andrzej Nagórko"
author = "Andrzej Nagórko"
# version = release = importlib.metadata.version("stackelberg-games-core")
version = release = "0.1"

extensions = [
    "sphinx_toolbox.more_autodoc.typevars",
    "sphinx.ext.autodoc",
    "sphinx.ext.duration",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.viewcode",
    "numpydoc",
    #    "sphinxcontrib.apidoc",
    "sphinx_exec_code",
    "sphinxcontrib.bibtex",
    "sphinx.ext.mathjax",
    "sphinx_math_dollar",
    "sphinx_proof",
    "sphinx.ext.todo",
    "matplotlib.sphinxext.plot_directive",
    "sphinxcontrib.datatemplates",
    "sphinx_toolbox.wikipedia",
]

templates_path = ["_templates"]
source_suffix = [".rst", ".md"]
exclude_patterns = [
    "_build",
    "*.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

autosummary_generate = True  # Turn on sphinx.ext.autosummary

html_theme = "pydata_sphinx_theme"
html_title = "Stackelberg Games Repository"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]
html_theme_options = {
    "logo": {
        "text": "AI for Security Research Group",
        # "image_light": "_static/rps_logo.png",
        # "image_dark": "_static/rps_logo.png",
    },
    "show_toc_level": 2,
    "pygments_light_style": "gotthard-light",
    "pygments_dark_style": "gotthard-dark",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# apidoc_module_dir = "../src/bsgmc"
# apidoc_output_dir = "api"
# apidoc_excluded_paths = ["tests"]
# apidoc_separate_modules = True

autodoc_member_order = "bysource"
autodoc_preserve_defaults = True
autodoc_typehints = "description"

copybutton_exclude = ".linenos, .gp, .go"

bibtex_bibfiles = ["whitepaper/references.bib"]
bibtex_reference_style = "label"

mathjax3_config = {
    "tex": {
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
        "macros": {
            "Prob": r"\text{Prob}",
            "conv": r"\text{conv}",
            "cone": r"\text{cone}",
            "argmax": r"\mathop{\rm argmax}",
            "abr": r"\mathop{\rm abr}\nolimits",
            "softmax": r"\text{softmax}",
            "relu": r"\text{ReLU}",
            "lin": r"\text{lin}",
            "span": r"\text{span}",
        },
    },
}

todo_include_todos = True