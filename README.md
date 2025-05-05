# EMMo

EMMo (**E**xpectation-**M**aximization-based **Mo**tif finder) is Python toolkit for training and
exploring unsupervised methods for:

- peptide-MHC binding motif identification,
- deconvolution of multi-allelic data, and
- peptide-MHC binding/presentation prediction.

The documentation can be found [here](https://instadeep.gitlab.io/emmo).

## Installation

Please refer to the following documentation page for installation steps:
[Installation](https://instadeep.gitlab.io/emmo/installation.html)

## Overview

After installation, you can list all available commands by running

```bash
emmo --help
```

For the full documentation of a specific command you can run:

```bash
emmo {command} --help
```

All the Python code of the project can be found in the [emmo](emmo) directory and the main
entrypoint is the file [emmo/main.py](emmo/main.py).

If you want to learn more about the implemented deconvolution and prediction methods, check out the
explanations for the
[MHC2 expectation-maximization algorithm](https://instadeep.gitlab.io/emmo/explanations/em_algorithm.html)
and the
[MHC2 binding predictors](https://instadeep.gitlab.io/emmo/explanations/prediction_mhc2.html).

For additional documentation (usage and explanations) you can check the
[project's documentation](https://instadeep.gitlab.io/emmo/). To contribute to this repository,
check the [Contributing](https://instadeep.gitlab.io/emmo/CONTRIBUTING.html) section.
