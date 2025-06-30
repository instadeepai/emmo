# MHC1 ligands in MHC2 ligandomics data

## Background

For some MHC2 mass spec experiments, motif deconvolution produces clusters that consist of MHC1
ligands. They can be identified by their typical MHC1 anchor positions and also by the corresponding
length distribution that peaks at 9-mers. An explanation for seeing MHC1 ligands in MHC2 ligandomics
data is the experimental procedure, in particular, the use of the same instruments and devices for
MHC1 and MHC2 experiments.

## Method description

EMMo contains functions to annotate this type of contamination in a semi-automated manner. A
challenge is to select the right $K$, i.e., the number of clusters/motifs that shall be identified
by the decovolution algorithm. In [MoDec](https://github.com/GfellerLab/MoDec)
[(Racle et al., 2019)](https://www.nature.com/articles/s41587-019-0289-6), the Akaike information
criterion (AIC) is used to identify the optimal $K$:

$$
\textrm{AIC} = 2k - \ln(\hat{L})
$$

where $k$ is the number of estimated parameters and $\hat{L}$ is the likelihood under the model.

However, we observed that the model with the best AIC often has too few motifs resulting in clusters
containing MHC1 and MHC2 ligands.

Therefore, EMMo implements the following heuristic strategy, which assumes that increasing $K$
preserves the identification of clusters that are dominated by 9-mers:

1. Perform deconvolution for every mass spectrometry experiment using a range of values for $K$. The
   minimal value for $K$ is naturally set to 1, while the maximal value, $K_{\max}$, must be
   determined by the user. Increasing $K_{\max}$ will result in longer run times and a higher
   likelihood of generating noisy motifs or clusters that may not correspond to HLA alleles,
   especially for experiments with fewer ligands. For single-allelic experiments, a lower $K_{\max}$
   (e.g., $K_{\max}=3$) is sufficient, whereas multi-allelic experiments require a higher
   $K_{\max}$.
1. For each experiment, identify the model $K^{*}$ with the best AIC.
1. For each experiment, peptide, and model $1 \le K \le K_{\max}$, annotate the peptide as
   _K-contaminant_ if
   - The peptide is part of a cluster with
     - Length mode $\le9$, or
     - Length mode $=10$ and offset shift $+1$ towards the C-terminus (typical for 9-mer motifs from
       10mer MHC1 ligands).
   - This cluster is not the flat motif.
   - The peptide length is $\le 12$.
1. Annotate a peptide in an experiment as _contaminant_ if the peptide is $K$-contaminant for every
   $K \ge K^{*}$.

## Usage in EMMo

The heuristic described above is implemented in the `emmo.dataset_processing.contaminants` module.
The function `annotate_mhc1_contaminants_in_mhc2_ligandomics` extracts the peptides from the
`responsibilities.csv` files in the model directories and annotates them to identify potential MHC1
contaminants. The following code snippet is an example of how to use it:

```python
from emmo.dataset_processing.contaminants import (
    annotate_mhc1_contaminants_in_mhc2_ligandomics
)
from emmo.io.output import find_deconvolution_results

# find deconvolution results for all experiments and all numbers of clusters (K)
# in the specified directory
df_models = find_deconvolution_results("path/to/per_experiment_deconvolution_results")

# annotate peptides in the responsibility files whether they are potential contaminants
# according the the heuristic strategy
df_annotated_all = annotate_mhc1_contaminants_in_mhc2_ligandomics(df_models=df_models)
```

The resulting DataFrame will contain the following columns (only a subset is listed here):

- `group` - the group identifier (e.g., the experiment ID) from the directory names
- `peptide` - the peptide sequence
- `number_of_classes_best_aic` - the best number of classes $K^*$ for the group according to AIC
- `mhc1_contaminant_best_aic` - `True` if the peptide was annotated as $K^*$-contaminant (see
  above), `False` otherwise
- `mhc1_contaminant_annotation_consensus` - `True` if the peptide was annotated as contaminant (see
  above), `False` otherwise
