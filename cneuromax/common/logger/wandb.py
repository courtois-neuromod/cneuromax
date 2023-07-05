"""."""

from hydra_zen import store


@store(name="wandb", group="logger")
class WandbLoggerConfig:
    """.

    Attributes:
        target_: Logger class.
        name: The name of the run.
        project: The name of the project to which this run will belong.
    """

    _target_: str = "lightning.pytorch.loggers.wandb.WandbLogger"
    name: str | None = None
    project: str | None = None
