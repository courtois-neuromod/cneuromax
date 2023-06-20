# CNeuroML

[![container-build-push](
    https://github.com/courtois-neuromod/cneuroml/actions/workflows/container-build-push.yaml/badge.svg)](
        https://github.com/courtois-neuromod/cneuroml/actions/workflows/container-build-push.yaml)
[![docs-build-push](
    https://github.com/courtois-neuromod/cneuroml/actions/workflows/docs-build-push.yaml/badge.svg)](
        https://github.com/courtois-neuromod/cneuroml/actions/workflows/docs-build-push.yaml)
[![format-lint](
    https://github.com/courtois-neuromod/cneuroml/actions/workflows/format-lint.yaml/badge.svg)](
        https://github.com/courtois-neuromod/cneuroml/actions/workflows/format-lint.yaml)
[![typecheck-unittest](
    https://github.com/courtois-neuromod/cneuroml/actions/workflows/typecheck-unittest.yaml/badge.svg)](
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
cneuroml
├─ .github
│  └─ workflows                                    <-- Config files launched by github actions (test format, rebuild container image, etc) 
│     ├─ container-build-push.yaml
│     ├─ container-build.yaml
│     ├─ docs-build-push.yaml
│     ├─ docs-build.yaml
│     ├─ format-lint.yaml
│     └─ typecheck-unittest.yaml
├─ .gitignore
├─ .pre-commit-config.yaml                         <-- Tests to check for errors before a commit
├─ .yamllint.yaml                                  <-- Config file for .yaml files 
├─ LICENSE
├─ README.md
├─ cneuroml                                        <-- Core ML scripts 
│  ├─ __init__.py
│  ├─ app
│  │  └─ __init__.py
│  ├─ dl                                           <-- Deep Learning scripts 
│  │  ├─ __init__.py
│  │  └─ base                                      <-- Base/generic/examplar scripts 
│  │     ├─ __init__.py
│  │  │  ├─ configs                                <-- Generic configuration files (.yaml)
│  │  │  ├─ datasets                               <-- Generic DataModules
│  │  │  │  ├─ datamodule.py
│  │  │  │  └─ datamodule_test.py
│  │  │  ├─ models                                 <-- Generic models (lightning modules)
│  │  │  │  ├─ litmodule.py
│  │  │  │  └─ litmodule_test.py
│  │  │  ├─ utils                                  <-- Generic utils scripts
│  │  │  ├─ eval.py                                <-- Generic eval script
│  │  │  └─ train.py                               <-- Generic train script
│  │  └─ projects                                  <-- Where to save your project scripts, per model and dataset (e.g., video_transformer/friends)
│  │     └─ my_model_type
│  │        └─ models                              <-- Project-specific models (lightning modules)
│  │        └─ my_model_and_dataset
│  │           ├─ __init__.py
│  │           ├─ configs                          <-- Project-specific configuration files (.yaml)
│  │           ├─ datasets                         <-- Project-specific DataModules
│  │           ├─ models                           <-- Project-specific models (if needed)
│  │           ├─ utils                            <-- Project-specific utils scripts
│  │           ├─ my_eval.py                       <-- Project-specific eval script (if needed)
│  │           └─ my_train.py                      <-- Project-specific train script (if needed)
│  └─ ne                                           <-- Neuro Evolution scripts 
│     └─ __init__.py
├─ containers
│  └─ deps
│     └─ run
│        └─ Containerfile                          <-- File to build container image 
├─ data                                            <-- Directory where to install datasets as datalad submodules  
├─ docs                                            <-- Documentation files and sphinx requirements
│  ├─ *.rst
│  ├─ Makefile
│  ├─ conf.py
│  ├─ make.bat
│  └─ requirements.txt
├─ pyproject.toml                                  <-- Library's general config file for anything python-related 
├─ pyreqs                                          <-- Python requirements for different project tools
│  └─ *.txt
└─ renovate.json                                   <-- Scans the web to keep config packages up-to-date  
