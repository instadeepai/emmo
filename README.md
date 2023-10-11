# EMMo

**E**xpectation-**M**aximization-based **Mo**tif finder.

## Description

Code repository for exploring unsupervised methods for:

- peptide-MHC binding motif identification,
- deconvolution of multi-allelic data, and
- peptide-MHC binding/presentation prediction.

The package contains Python implementations of the methods MixMHCp for MHC1 deconvolution (Gfeller
et al. 2018, J Immunol), MoDec for MHC2 deconvolution (Racle et al. 2019, Nat Biotechnol), and
MixMHC2pred for MHC2 binding prediction (Racle et al. 2019, Nat Biotechnol), with some
modifications, simplifications, and improvements.

## Installation

#### Requirements

- [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) - 23.5.2

#### Step-by-step

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

1. _[Only for dev changes]_ Refer to the [contributing document](../CONTRIBUTING.md) for further
   contributing guidelines for the project.
