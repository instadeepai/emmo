# EMMo

EMMo (**E**xpectation-**M**aximization-based **Mo**tif finder) is Python toolkit for training and
exploring unsupervised methods for:

- peptide-MHC binding motif identification,
- deconvolution of multi-allelic data, and
- peptide-MHC binding/presentation prediction.

The package contains Python implementations of the methods MixMHCp for MHC1 deconvolution
[(Gfeller et al., 2018)](https://journals.aai.org/jimmunol/article/201/12/3705/106932/The-Length-Distribution-and-Multiple-Specificity),
MoDec for MHC2 deconvolution
[(Racle et al., 2019)](https://www.nature.com/articles/s41587-019-0289-6), and MixMHC2pred for MHC2
binding prediction [(Racle et al., 2019)](https://www.nature.com/articles/s41587-019-0289-6), with
some modifications, simplifications, and improvements.

## Overview

Please following the [installation guidelines](installation.md) to install EMMo.

After installation, you can list all available commands by running

```bash
emmo --help
```

For the full documentation of a specific command you can run:

```bash
emmo {command} --help
```

To contribute to this repository, check the [Contributing](CONTRIBUTING.md) section.
