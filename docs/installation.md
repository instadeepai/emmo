# Installation

## Users

To install EMMo, you can run

```
pip install .
```

Precompiled models for peptide-MHC2 binding prediction can be downloaded from the GCP bucket to the
folder `models/binding_predictor`. These models can then be loaded in the `emmo predict-mhc2`
command-line script by simply providing their name instead of a full path, under the condition that
the `models` folder is in the correct relative location w.r.t. the package installation (i.e., the
same as in the repository). An option to ensure the latter is to download the complete repository
and install the package in editable mode. To this end, execute the following command in the root
directory of the repository.

```
pip install -e .
```

## Developers

### Requirements

- [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) - 23.5.2

### Step-by-step

1. Clone the repository:

   ```bash
   git clone git@gitlab.com:instadeep/emmo.git && cd emmo
   ```

1. _[Only for dev changes]_ Create and activate the conda environment named `emmo_pre_commit`:

   ```bash
   conda env create -f .pre-commit-env.yaml && conda activate emmo_pre_commit
   ```

1. _[Only for dev changes]_ Install the pre-commit (make sure to have the conda environment
   activated)

   ```bash
   pre-commit install -t pre-commit -t commit-msg
   ```

1. _[Only for dev changes]_ Refer to the [contributing document](CONTRIBUTING.md) for further
   contributing guidelines for the project.

### Setup remote files support

1. Copy `.env.template` to `.env`

   ```bash
   cp .env.template .env
   ```

1. Copy json file with GCP credentials to `.credentials/biontech-tcr-16ca4aceba4c.json` (same file
   as the one used in BioNDeep).
