"""."""

from dataclasses import dataclass
from typing import Annotated as An

from omegaconf import DictConfig

from cneuromax.utils.annotations import not_empty, one_of


@dataclass
class BaseFitterHydraConfig(DictConfig):
    """Root Fitter Hydra configuration.

    Attributes:
        data_dir: .
        device: Computing device to use for large matrix operations.
        load_path: Path to the model to load.
        load_path_pbt: Path to the HPO checkpoint to load for PBT.
        save_path: Path to save the model.
        copy_data_commands: Commands to copy data to the cluster.
    """

    data_dir: An[str, not_empty()] = "data/example_run/"
    device: An[str, one_of("cpu", "gpu")] = "cpu"
    load_path: An[str, not_empty()] | None = None
    load_path_pbt: An[str, not_empty()] | None = None
    save_path: An[str, not_empty()] = "${data_dir}/lightning/final.ckpt"
    copy_data_commands: list[str] | None = None
