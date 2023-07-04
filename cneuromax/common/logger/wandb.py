"""."""
from hydra_zen import hydrated_dataclass, store
from lightning.pytorch.loggers.wandb import WandbLogger


@store(name="wandb", group="logger")
@hydrated_dataclass(target=WandbLogger)
class WandbLoggerConfig:
    """."""

    name: str | None = None
    project: str | None = None
