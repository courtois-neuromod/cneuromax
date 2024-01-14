""":class:`BaseSubtaskConfig` & :class:`BaseHydraConfig`.

Check-out the `hydra docs \
<https://hydra.cc/docs/tutorials/structured_config/intro/>`_
& `omegaconf docs \
<https://omegaconf.readthedocs.io/en/2.1_branch/structured_config.html>`_
for more information on how structured configurations work and how to
best utilize them.

This module also makes use of :func:`hydra_zen.make_config` to
simplify config creation (`reference \
<https://mit-ll-responsible-ai.github.io/hydra-zen/generated/\
hydra_zen.make_config.html>`_).
"""
from dataclasses import dataclass
from typing import Annotated as An

from hydra import conf as hc
from hydra import types as ht
from hydra.experimental.callbacks import LogJobReturnCallback
from hydra_zen import make_config

from cneuromax.utils.beartype import not_empty
from cneuromax.utils.hydra_zen import fs_builds


@dataclass
class BaseSubtaskConfig:
    """Base ``subtask`` config.

    Args:
        output_dir: Path to the ``subtask`` output directory. Every\
            artifact generated during the ``subtask`` will be stored\
            in this directory.
        data_dir: Path to the data directory. This directory is\
            shared between ``task`` runs. It is used to store\
            datasets, pre-trained models, etc.
    """

    output_dir: An[str, not_empty()] = "${hydra:runtime.output_dir}"
    data_dir: An[str, not_empty()] = "${oc.env:CNEUROMAX_PATH}/data/"


@dataclass
class BaseHydraConfig(
    make_config(  # type: ignore[misc]
        bases=(hc.HydraConf,),
        callbacks={"log_job_return": fs_builds(LogJobReturnCallback)},
        job=hc.JobConf(
            config=hc.JobConf.JobConfig(
                override_dirname=hc.JobConf.JobConfig.OverrideDirname(
                    kv_sep=".",
                    item_sep="~",
                    exclude_keys=["task"],
                ),
            ),
        ),
        mode=ht.RunMode.MULTIRUN,
        sweep=hc.SweepDir(
            dir="${oc.env:CNEUROMAX_PATH}/data/test/",
            subdir="${hydra:job.override_dirname}",
        ),
    ),
):
    """Base :mod:`hydra.conf.HydraConf` config."""
