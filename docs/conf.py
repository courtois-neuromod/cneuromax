"""Configuration file for the Sphinx documentation builder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path("..").resolve()))

# -- Project information

project = "CNeuroML"
copyright = "2023, The CNeuroML Authors"  # noqa: A001
author = "The CNeuroML Authors"
version = "0.0.1"

# -- General configuration

extensions = [
    "autoapi.extension",
    "sphinxcontrib.autoyaml",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
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
autoapi_dirs = ["../cneuroml"]
autoapi_python_class_content = "both"
