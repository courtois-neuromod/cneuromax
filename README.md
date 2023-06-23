# CNeuroML

[![container-build-push](
    https://github.com/courtois-neuromod/cneuroml/actions/workflows/container-build-push.yaml/badge.svg)](
        https://github.com/courtois-neuromod/cneuroml/actions/workflows/container-build-push.yaml)
[![docs-build-push](
    https://github.com/courtois-neuromod/cneuroml/actions/workflows/docs-build-push.yaml/badge.svg)](
        https://github.com/courtois-neuromod/cneuroml/actions/workflows/docs-build-push.yaml)
[![format-lint](
    https://github.com/courtois-neuromod/cneuroml/actions/workflows/format-lint.yaml/badge.svg?event=push)](
        https://github.com/courtois-neuromod/cneuroml/actions/workflows/format-lint.yaml)
[![typecheck-unittest](
    https://github.com/courtois-neuromod/cneuroml/actions/workflows/typecheck-unittest.yaml/badge.svg?event=push)](
        https://github.com/courtois-neuromod/cneuroml/actions/workflows/typecheck-unittest.yaml)
[![codecov](
    https://codecov.io/gh/courtois-neuromod/cneuroml/branch/main/graph/badge.svg?token=AN8GLFP9CB)](
        https://codecov.io/gh/courtois-neuromod/cneuroml)
[![code style: black](
    https://img.shields.io/badge/code%20style-black-000000.svg)](
        https://github.com/psf/black)

Full documentation available at [https://courtois-neuromod.github.io/cneuroml](
    https://courtois-neuromod.github.io/cneuroml/).


```
cneuroml/
├─ .github/                     <-- Contains GitHub automation (tests, containers, etc) config files
│  └─ *
├─ cneuroml/                    <-- Contains all of the Machine Learning code and config files
│  ├─ application/              <-- Contains the code to create applications (cozmo inference, etc)
│  │  └─ *
│  ├─ deeplearning/             <-- Contains the Deep Learning code
│  │  ├─ common/                <-- Contains the code common to more than one DL experiment
│  │  │  ├─ datamodule/         <-- Contains common Lightning DataModules
│  │  │  │  ├─ base.py          <-- The base Lightning DataModule to build upon
│  │  │  │  └─ *
│  │  │  ├─ litmodule/          <-- Contains common Lightning Modules
│  │  │  │  ├─ base.py          <-- The base Lightning Module to build upon
│  │  │  │  └─ *
│  │  │  └─ nnmodule/           <-- Contains common PyTorch Modules
│  │  │     └─ *
│  │  ├─ experiments/           <-- Contains the DL experiments
│  │  │  ├─ my_new_experiment/  <-- ! Your new DL experiment folder
│  │  │  │  ├─ datamodule.py    <-- ! Your Lightning DataModule
│  │  │  │  ├─ litmodule.py     <-- ! Your Lightning Module
│  │  │  │  ├─ nnmodule.py      <-- ! Your PyTorch Module
│  │  │  │  ├─ config.yaml      <-- ! Your Hydra configuration file
│  │  │  │  └─ *
│  │  │  └─ *
│  │  ├─ fitter.py              <-- Contains the code to fit (train/val/test) a DL model
│  │  └─ *
│  └─ neuroevolution/           <-- Contains the code for Neuroevolution models
│     └─ *
├─ containers/                  <-- Contains the Containerfiles to build the Podman/Docker images
│  └─ *
├─ data/                        <-- Directory in which to download datasets
│  └─ *
├─ docs/                        <-- Contains the documentation files
│  └─ *
├─ pyreqs/                      <-- Python requirements for different experiments
│  └─ *
├─ .gitignore                   <-- Files to not track with Git/GitHub
├─ .pre-commit-config.yaml      <-- Pre git commit configuration (for formatting, linting, etc)
├─ .yamllint.yaml               <-- YAML files configuration
├─ LICENSE                      <-- MIT License file
├─ README.md                    <-- Repository description file
├─ pyproject.toml               <-- Python files configuration
└─ renovate.json                <-- Renovate Bot (keeps dependencies up-to-date) configuration
