"""Configuration file for the Sphinx documentation builder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path("..").resolve()))
suppress_warnings = ["*"]
project = "CNeuroMax"
copyright = "2023, The CNeuroMax Authors"  # noqa: A001
author = "The CNeuroMax Authors"
version = "0.0.1"
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_paramlinks",
]
autodoc_default_options = {"show-inheritance": True}
autodoc_member_order = "bysource"
autosummary_generate = True
html_theme = "furo"
html_title = "CNeuroMax"
paramlinks_hyperlink_param = "name"
templates_path = ["_templates"]
typehints_defaults = "comma"
