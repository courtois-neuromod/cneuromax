"""Configuration file for the Sphinx documentation builder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path("..").resolve()))
project = "CNeuroMax"
copyright = "2023, The CNeuroMax Authors"  # noqa: A001
author = "The CNeuroMax Authors"
version = "0.0.1"

# -- General configuration

extensions = [
    "autoapi.extension",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.duration",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_paramlinks",
]
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
autoapi_template_dir = "_autoapi_templates"
autoapi_type = "python"
html_theme = "furo"
html_title = "CNeuroMax"
