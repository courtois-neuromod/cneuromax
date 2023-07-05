"""."""

from typing import ClassVar

from hydra_zen import store

from cneuromax.common.launcher.local import LocalLauncherConfig


@store(name="slurm", group="launcher")
class BaseSlurmLauncherConfig(LocalLauncherConfig):
    """.

    Attributes:
        target_: .
        account: .
        setup: Setup steps to run before the job (load the container)
    """

    _target_: str = (
        "hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher"
    )
    account: str = "rrg-pbellec"
    setup: ClassVar[list[str]] = [
        "module load podman/4.5.0",
        "nvidia-ctk cdi generate --output=/var/tmp/cdi/nvidia.yaml",
        "cp ${SCRATCH}/container.tar ${SLURM_TMPDIR}/${SCRATCH}/.",
        "tar -xf ${SLURM_TMPDIR}/${SCRATCH}/container.tar",
        "-C ${SLURM_TMPDIR}/${SCRATCH}/.",
    ]
