"""Base Deep Learning Module.

This module contains elements common to more than one Deep Learning
component in the CNeuroML library.
"""

from cneuroml.dl.base.datamodule import BaseDataModule
from cneuroml.dl.base.litmodule import BaseLitModule

__all__ = ["BaseDataModule", "BaseLitModule"]
