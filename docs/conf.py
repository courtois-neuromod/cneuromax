"""Configuration file for the Sphinx documentation builder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path("..").resolve()))

# -- Project information

project = "cneuromax"
copyright = "2023, The cneuromax Authors"  # noqa: A001
author = "The cneuromax Authors"
version = "0.0.1"

# -- General configuration

extensions = [
    "autoapi.extension",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_paramlinks",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "furo"

autoapi_type = "python"
autoapi_dirs = ["../cneuromax"]
autoapi_keep_files = True
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-module-summary",
    "special-members",
    "imported-members",
]
