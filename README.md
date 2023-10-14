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

Full documentation available at [https://courtois-neuromod.github.io/cneuromax](
    https://courtois-neuromod.github.io/cneuromax/).


```
cneuromax/
├─ .github/                  <-- Config files for GitHub automation (tests, containers, etc)
├─ cneuromax/                <-- Machine Learning code
│  ├─ fitting/               <-- ML model fitting code
│  │  ├─ common/             <-- Code common to all fitting workflows
│  │  │  ├─ __init__.py      <-- Stores common Hydra configs
│  │  │  └─ fitter.py        <-- Base Hydra config common to all fitting workflows
│  │  ├─ deeplearning/       <-- Deep Learning code
│  │  │  ├─ datamodule/      <-- Lightning DataModules
│  │  │  │  ├─ base.py       <-- Base Lightning DataModule to build upon
│  │  │  ├─ litmodule/       <-- Lightning Modules
│  │  │  │  ├─ base.py       <-- Base Lightning Module to build upon
│  │  │  ├─ nnmodule/        <-- PyTorch Modules & Hydra configs
│  │  │  ├─ utils/           <-- Deep Learning utilities
│  │  │  ├─ __init__.py      <-- Stores Deep Learning Hydra configs
│  │  │  ├─ __main__.py      <-- Entrypoint when calling `python cneuromax.fitting.deeplearning`
│  │  │  ├─ config.yaml      <-- Default Hydra configs & settings
│  │  │  └─ fitter.py        <-- Deep Learning fitting
│  │  └─ neuroevolution/     <-- Neuroevolution code
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
│  └─ utils/                 <-- CNeuroMax utilities
├─ docs/                     <-- Documentation files
├─ .devcontainer.json        <-- VSCode container development config
├─ .gitignore                <-- Files to not track with Git/GitHub
├─ .pre-commit-config.yaml   <-- Pre-"git commit" actions config (format, lint, etc)
├─ .yamllint.yaml            <-- YAML files config
├─ Containerfile                <-- To build the Docker image
├─ LICENSE                   <-- MIT License file
├─ README.md                 <-- Repository description file
├─ pyproject.toml            <-- Python code & dependencies config
└─ renovate.json             <-- Renovate Bot config (keeps dependencies up-to-date)
```
