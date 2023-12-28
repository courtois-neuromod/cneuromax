# CNeuroMax

[![container-build-push](
    https://github.com/courtois-neuromod/cneuromax/actions/workflows/container-build-push.yaml/badge.svg)](
        https://github.com/courtois-neuromod/cneuromax/actions/workflows/container-build-push.yaml)
[![docs-build-push](
    https://github.com/courtois-neuromod/cneuromax/actions/workflows/docs-build-push.yaml/badge.svg)](
        https://github.com/courtois-neuromod/cneuromax/actions/workflows/docs-build-push.yaml)
[![format-lint](
    https://github.com/courtois-neuromod/cneuromax/actions/workflows/format-lint.yaml/badge.svg?event=push)](
        https://github.com/courtois-neuromod/cneuromax/actions/workflows/format-lint.yaml)
[![typecheck-unittest](
    https://github.com/courtois-neuromod/cneuromax/actions/workflows/typecheck-unittest.yaml/badge.svg?event=push)](
        https://github.com/courtois-neuromod/cneuromax/actions/workflows/typecheck-unittest.yaml)
[![codecov](
    https://codecov.io/gh/courtois-neuromod/cneuromax/branch/main/graph/badge.svg?token=AN8GLFP9CB)](
        https://codecov.io/gh/courtois-neuromod/cneuromax)
[![code style: black](
    https://img.shields.io/badge/code%20style-black-000000.svg)](
        https://github.com/psf/black)

<h2>Overview</h2>

CNeuroMax is a machine learning
([Deep Learning](https://en.wikipedia.org/wiki/Deep_learning) +
[Neuroevolution](https://en.wikipedia.org/wiki/Neuroevolution))
and serving ([Lightning Apps](https://lightning.ai/docs/app/stable/))
workspace. CNeuroMax aims to:

**1. Accelerate machine learning processes:**
* Hyperparameter optimization with [Orion](https://github.com/Epistimio/orion)'s
[Hydra Sweeper plugin](https://github.com/Epistimio/hydra_orion_sweeper).
* SLURM job definition, queuing and monitoring with
[Submitit](https://github.com/facebookincubator/submitit)'s
[Hydra Launcher plugin](https://hydra.cc/docs/plugins/submitit_launcher/).
* [Docker](https://www.docker.com/) / [Podman](https://podman.io/) /
[Apptainer](https://apptainer.org/) environment containerization for both
regular & SLURM-based execution.
* SLURM login-node [Apptainer](https://apptainer.org/)-backed Python commands
to launch training jobs.

**2. Automate workspace & coding processes:**
* Package upgrades through
[Renovate](https://github.com/renovatebot/renovate).
* Docstring documentation auto-generation with [Sphinx AutoAPI](https://github.com/readthedocs/sphinx-autoapi).
* Pre-commit formatting & linting hooks with
[pre-commit](https://pre-commit.com/).
* Documentation/Docker image validation/deployment, formatting, linting,
type-checking & unit tests upon contribution to the ``main`` branch with
[GitHub Actions](https://github.com/features/actions).

**3. Reduce code boilerplate with:**
* [Hydra](https://github.com/facebookresearch/hydra) for task/experiment
configuration.
* [Hydra-zen](https://github.com/mit-ll-responsible-ai/hydra-zen) for
[Hydra](https://github.com/facebookresearch/hydra) structured configuration
management.
* [Lightning](https://github.com/Lightning-AI/pytorch-lightning) for
[PyTorch](https://github.com/pytorch/pytorch) code.

**4. Smoothen researcher collaboration through:**
* An object-oriented code structure for code sharing & reusability.
* A mono-repository workspace task/experiment-specific subdirectories.
* A [Weights & Biases](https://wandb.ai/site) team for shared logging.

**5. Facilitate high-quality and reproducible code by:**
* Linting with [Ruff](https://github.com/astral-sh/ruff).
* Formatting with [Black](https://github.com/psf/black).
* Unit-testing with [pytest](https://github.com/pytest-dev/pytest).
* Type-checking statically with [Mypy](https://github.com/python/mypy)
& dynamically with [Beartype](https://github.com/beartype/beartype).
* [PyTorch](https://github.com/pytorch/pytorch) tensor &
[NumPy](https://github.com/numpy/numpy) array dtype & shape type hints
through [jaxtyping](https://github.com/google/jaxtyping) and
[nptyping](https://github.com/ramonhagenaars/nptyping) that are validatable
with [Beartype](https://github.com/beartype/beartype).
* Providing a common [Development Container](https://containers.dev/)
recipe with the above features enabled + automatic documentation preview
with [esbonio](https://github.com/swyddfa/esbonio) +
[GitHub Copilot](https://github.com/features/copilot).

**6. Smoothen up rough edges by providing:**
* Extensive documentation on how to install/execute the code on local/remote
& SLURM-based systems.
* Unassuming guides on how to contribute to the codebase.
* Tutorials on how to facilitate code transport across machines & prune
unnecessary components for paper publication.
* Offline [Weights & Biases](https://wandb.ai/site) support with
[wandb-osh](https://github.com/klieret/wandb-offline-sync-hook).

<h2>High-level repository tree:</h2>

```
cneuromax/
├─ .github/                  <-- Config files for GitHub automation (tests, containers, etc)
├─ cneuromax/                <-- Machine Learning code
│  ├─ fitting/               <-- ML model fitting code
│  │  ├─ deeplearning/       <-- Deep Learning code
│  │  │  ├─ datamodule/      <-- Lightning DataModules
│  │  │  ├─ litmodule/       <-- Lightning Modules
│  │  │  ├─ nnmodule/        <-- PyTorch Modules
│  │  │  ├─ utils/           <-- Deep Learning utilities
│  │  │  ├─ __init__.py      <-- Stores Deep Learning Hydra configs
│  │  │  ├─ __main__.py      <-- Entrypoint when calling `python cneuromax.fitting.deeplearning`
│  │  │  ├─ config.py        <-- Deep Learning structured Hydra config & utilities
│  │  │  ├─ config.yaml      <-- Default Deep Learning Hydra configs & settings
│  │  │  └─ fit.py           <-- Deep Learning fitting function
│  │  ├─ hybrid/             <-- Hybrid Deep Learning + Neuroevolution code
│  │  │  ├─ __init__.py      <-- Stores Hybrid DL + NE Hydra configs
│  │  │  ├─ __main__.py      <-- Entrypoint when calling `python cneuromax.fitting.hybrid`
│  │  │  ├─ config.py        <-- Hybrid DL + NE structured Hydra config & utilities
│  │  │  ├─ config.yaml      <-- Default Hybrid DL + NE Hydra configs & settings
│  │  │  └─ fit.py           <-- Hybrid DL + NE fitting function
│  │  ├─ neuroevolution/     <-- Neuroevolution code
│  │  │  ├─ agent/           <-- Neuroevolution agents (encapsulate networks)
│  │  │  ├─ net/             <-- Neuroevolution networks
│  │  │  ├─ space/           <-- Neuroevolution spaces (where agents get evaluated)
│  │  │  ├─ utils/           <-- Neuroevolution utilities
│  │  │  ├─ __init__.py      <-- Stores Neuroevolution Hydra configs
│  │  │  ├─ __main__.py      <-- Entrypoint when calling `python cneuromax.fitting.neuroevolution`
│  │  │  ├─ config.py        <-- Neuroevolution structured Hydra config & utilities
│  │  │  ├─ config.yaml      <-- Default Neuroevolution Hydra configs & settings
│  │  │  └─ fit.py           <-- Neuroevolution fitting function
│  │  ├─ __init__.py         <-- Stores Fitting Hydra configs
│  │  ├─ __main__.py         <-- Entrypoint when calling `python cneuromax.fitting`
│  │  └─ config.py           <-- Base Structured Hydra fitting config & utilities
│  ├─ serving/               <-- Contains the code to create applications (cozmo inference, etc)
│  ├─ task/                  <-- Contains the Deep Learning tasks
│  │  │
│  │  │                          ******************************************
│  │  └─ my_new_task/        <-- *** Your new Deep Learning task folder ***
│  │     ├─ __init__.py      <-- ********** Your Hydra Configs ************
│  │     ├─ datamodule.py    <-- ******* Your Lightning DataModule ********
│  │     ├─ litmodule.py     <-- ********* Your Lightning Module **********
│  │     ├─ nnmodule.py      <-- ********** Your PyTorch Module ***********
│  │     └─ config.yaml      <-- ****** Your Hydra configuration file *****
│  │                             ******************************************
│  │
│  ├─ utils/                 <-- CNeuroMax utilities
│  ├─ __init__.py            <-- Sets up Beartype
│  └─ config.py              <-- Base Structured Hydra config & utilities
├─ docs/                     <-- Documentation files
├─ .devcontainer.json        <-- VSCode container development config
├─ .gitignore                <-- Files to not track with Git/GitHub
├─ .pre-commit-config.yaml   <-- Pre-"git commit" actions config (format, lint, etc)
├─ .yamllint.yaml            <-- YAML files config
├─ Dockerfile             <-- To build the Docker image
├─ LICENSE                   <-- MIT License file
├─ README.md                 <-- Repository description file
├─ pyproject.toml            <-- Python code & dependencies config
└─ renovate.json             <-- Renovate Bot config (keeps dependencies up-to-date)
```

<h2>Additional information:</h1>

CNeuroMax is developed in the context of the
[Courtois Project on Neuronal Modelling](https://cneuromod.ca), also known as
CNeuroMod. Launched in 2018, CNeuroMod aims to create more human-like AI models
by training them to emulate both human brain dynamics and behaviour. Phase I
(2018-2023) of the project saw large-scale data acquisition and preliminary
unimodal modelling. Phase II (2023-2027) of the project aims to create
multimodal phantom models of the subjects. CNeuroMax is used as the framework
to train these phantom models.
