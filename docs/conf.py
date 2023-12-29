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
    "sphinx_copybutton",
    "sphinx_paramlinks",
]
add_module_names = False
autosummary_generate = True
paramlinks_hyperlink_param = "name"
html_theme = "furo"
html_title = "CNeuroMax"
templates_path = ["_templates"]
