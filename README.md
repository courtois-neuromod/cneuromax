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


    Project Organization
    ------------

        ├── LICENSE
        ├── README.md
        ├── pyproject.toml                  <- Project config and dependencies
        ├── renovate.json                   <- ...
        ├── cneuroml                        <- Main code repository
        │   └── app        
        │   └── dl        
        │   │   └── base    
        │   │       └── datamodule.tsv      <- Base code for Lightning DataModule       
        │   │       └── litmodule.tsv       <- Base code for Lightning Module             
        │   └── ne                        
        ├── containers                      
        │   └── deps  
        │       └── run        
        │           └── Contrainerfile     <- ...   
        ├── docs
        │   └── *.rst                       <- Various documentation pages on installation, development, execution, etc
        │   └── Makefile                    <- Makefile for Sphinx documentation
        │   └── conf.py                     <- Configuration file for the Sphinx documentation builder
        │   └── make.bat                    <- Command to build Sphinx documentation
        │   └── requirements.txt            <- Requirements to build Sphinx documentation
        │
        └── pyreqs                          <- Required packages
            └── core.txt                    <- Packages required to run any component of the CNeuroML library
            └── dl.txt                      <- Packages required to run the Deep Learning  component of the CNeuroML library
            └── hpo.txt                     <- Packages required to run Hyperparameter Optimization
            └── hydra_run.txt               <- Hydra used when running the code
            └── hydra_test.txt              <- Hydra used when testing the code
            └── jupyter.txt                 <- Packages required to work with Jupyter notebooks   
            └── ne.txt                    <- Packages required to run the Neuroevolution component of the CNeuroML library            

    --------
