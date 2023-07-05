"""."""

from dataclasses import dataclass

from hydra_zen import store


@store(name="local", group="launcher")
@dataclass
class LocalLauncherConfig:
    """.

    Attributes:
        _target_: .
        cpus_per_task: .
        num_gpus_per_node: .
        num_mem_gb: .
        name: .
        num_nodes: .
        submitit_folder: .
        tasks_per_node: .
        timeout_min: Job length in minutes.
    """

    _target_: str = (
        "hydra_plugins.hydra_submitit_launcher.submitit_launcher.LocalLauncher"
    )
    cpus_per_task: int = int("${num_cpus_per_task}")
    gpus_per_node: int = int("${num_gpus_per_node}")
    mem_gb: int = int("${num_mem_gb}")
    name: str = "${hydra.job.name}"
    nodes: int = int("${num_nodes}")
    submitit_folder: str = "${hydra.sweep.dir}/.submitit/%j"
    tasks_per_node: int = int("${num_tasks_per_node}")
    timeout_min: int = int("${num_mins}")
