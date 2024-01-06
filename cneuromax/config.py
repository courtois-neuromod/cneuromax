""":class:`.BaseSubtaskConfig` & :class:`.BaseHydraConfig`.

Check-out the `hydra docs \
<https://hydra.cc/docs/tutorials/structured_config/intro/>`_
& `omegaconf docs \
<https://omegaconf.readthedocs.io/en/2.1_branch/structured_config.html>`_
for more information on how structured configurations work and how to
best utilize them.
"""
from dataclasses import dataclass, field
from typing import Annotated as An

from hydra.conf import HydraConf, JobConf, SweepDir
from hydra.types import RunMode

from cneuromax.utils.annotations import not_empty


@dataclass
class BaseSubtaskConfig:
    """Base ``subtask`` config.

    See :meth:`~.BaseTaskRunner.store_configs` to see how this config
    is being stored.

    Args:
        output_dir: Path to the ``subtask`` output directory. Every\
            artifact generated during the ``subtask`` will be stored\
            in this directory.
        data_dir: Path to the data directory. This directory is\
            shared between :mod:`cneuromax` ``task`` runs. It is used\
            to store datasets, pre-trained models, etc.
    """

    output_dir: An[str, not_empty()] = "${hydra:runtime.output_dir}"
    data_dir: An[str, not_empty()] = "${oc.env:CNEUROMAX_PATH}/data/"


JobConfig = JobConf.JobConfig
OverrideDirname = JobConfig.OverrideDirname


@dataclass
class BaseHydraConfig(HydraConf):
    """Base :mod:`hydra.conf.HydraConf` config.

    See :meth:`~.BaseTaskRunner.store_configs` to see how this config
    is being stored.
    """

    @dataclass
    class BaseHydraJobConfig(JobConf):
        """Base :mod:`hydra.conf.HydraConf` job config."""

        @dataclass
        class BaseHydraJobConfigConfig(JobConfig):
            """Base :mod:`hydra.conf.HydraConf` job config's config."""

            override_dirname: OverrideDirname = field(
                default_factory=lambda: OverrideDirname(
                    kv_sep=".",
                    item_sep="~",
                    exclude_keys=["task"],
                ),
            )

    project_name: str = "project"
    task_name: str = "task"
    job: BaseHydraJobConfig = field(default_factory=BaseHydraJobConfig)
    mode: RunMode = RunMode.MULTIRUN
    sweep: SweepDir = field(
        default_factory=lambda: SweepDir(
            dir="${oc.env:CNEUROMAX_PATH}/data/test/",
            # "${hydra:project_name}/${hydra:task_name}",
            subdir="${hydra:job.override_dirname}",
        ),
    )
