# Peptides from unexpected alleles

## Background

MHC ligandomics data usually comes with the information on which alleles have been present in the
cell line or sample (or which alleles have been pulled down). Ideally, each obtained peptide would
have been bound to one of those alleles. However, for some mass spec experiments, we observe motifs
from alleles that should not have been present. The corresponding peptides can also be considered as
contaminants.

Ligands from contaminant alleles have been observed in single-allelic MHC1 data, especially for
certain cell lines. A typical method to generate single-allelic data is to re-introduce the allele
of interest into a cell line that was genetically engineered to not express any MHC alleles. In some
cases, the original alleles of the cell lines still seem to be present:

- Cell line **C1R** is known to have minimal HLA-B\*35:03 and normal levels of HLA-C\*04:01 cell
  surface expression (see e.g. [this publication](https://doi.org/10.3389/fimmu.2021.653710)).
- Cell line B721.221 supposedly does not express any HLA-I alleles (see
  [this website](https://www.cellosaurus.org/CVCL_6263)). However, we observe contaminant motifs
  that appear to be HLA-C\*01:02. This allele is among the HLA-I alleles of the parent cell line
  [LCL 721](https://www.cellosaurus.org/CVCL_2102) of B721.221.

For example, here is the deconvolution result ($K=2$) for a single-allelic experiment for
HLA-A\*11:01 where C1R was used. While motif 1 is the motif of the expected allele HLA-A\*11:01,
there is a clear second motif for HLA-C\*04:01.

<img src="../../media/usage/contaminant-C0401.png" width="800"/>

Increasing $K$ to $3$ usually produces a third motif corresponding to the second allele,
HLA-B\*35:03, that is still expressed in C1R.

<img src="../../media/usage/contaminant-C0401-B3503.png" width="800"/>

## Method description

We follow a simple heuristic strategy to identify peptides from unexpected alleles:

- We define a set of reference motifs for individual alleles (e.g., by manual curation of the motifs
  from deconvolution runs on single-allelic mass spec data
- We define a set $C$ of contaminant types to identify, including
  - HLA-C\*04:01 in C1R
  - HLA-B\*35:03 in C1R
  - HLA-C\*01:02 in B721
  - ...
- Given a peptide for a mass spec experiment for which contaminant types $C' \subseteq C$ must be
  considered, we apply likelihood scoring with
  - the original model of their deconvolution run (for simplicity, we use $K=1$), giving
    likelihood/score $S$, and
  - for each contaminant type $c\in C'$, a model consisting only of the motif of the respective
    contaminant allele (e.g., the motif obtained from HLA-C\*04:01 experiments), giving
    likelihood/score $S_c$
- If $S_c > S$ for the peptide and some $c\in C'$, then this peptide is annotated as contaminant

## Usage in EMMo

The semi-automated annotation of contaminant peptides from unexpected alleles requires the
configuration of per-allele reference motifs and the contaminant types.

### Per-allele reference motifs

See documentation page [Reference motifs](../reference_motifs.md).

### Contaminant types

The contaminant types to be identified can also be defined in a YAML file. Each type needs:

- `identifier` - a unique identifier
- `column` - the column in the experiments table to decide whether a contaminant type is relevant
  for a specific experiment (can be "cell_type", "allele", "experiment_id", etc.)
- `value` - the value to filter for in `column`
- `value_match` - how to match `value` with the entries in `column` (default: "equals")
- `contaminant_allele` - the contaminant allele in compact format

Additionally, and `exclude` block can be defined to define exclusion rules. For example,
single-allelic experiments for HLA-C\*04:01 that were conducted using C1R should be excluded from
identifying HLA-C\*04:01 contaminant peptides.

<details>
  <summary>Example config: Contaminant types</summary>

```yaml
contaminant_alleles:
  # cell line C1R, allele C*04:01
  - identifier: C1R_C0401
    column: cell_type
    value: C1R
    value_match: contains_insensitive
    contaminant_allele: C0401
    exclude:
      column: allele
      value: C0401
      value_match: equals
  # cell line C1R, allele B*35:03
  - identifier: C1R_B3503
    column: cell_type
    value: C1R
    value_match: contains_insensitive
    contaminant_allele: B3503
    exclude:
      column: allele
      value: B3503
      value_match: equals
  # cell line B721.221, allele C*01:02
  - identifier: B721_C0102
    column: cell_type
    value: B721
    value_match: contains_insensitive
    contaminant_allele: C0102
    exclude:
      column: allele
      value: C0102
      value_match: equals
  # experiment expE00970, allele B*37:01
  - identifier: expE00970_B3701
    column: experiment_id
    value: expE00970
    value_match: equals
    contaminant_allele: B3701
```

</details>

### Annotation

Peptides are finally annotated using the function `annotate_peptides_from_unexpected_alleles`. The
following inputs are required:

- per-experiment deconvolution runs directory (containing the peptides to annotate in the
  `responsibilities.csv` files)
- experiments table (containing columns `experiment_id` to merge deconvolution runs + any column
  specified in the contaminant types config file)
- reference per-allele deconvolution runs
- corresponding motif selection config (see above)
- contaminant types config file (see above)

```python
from emmo.dataset_processing.contaminants import annotate_peptides_from_unexpected_alleles

df_models, df_peptides = annotate_peptides_from_unexpected_alleles(
    path_deconvolution_runs="path/to/per_experiment_deconv_directory",
    path_experiments_table="path/to/experiment_table.csv",
    path_deconvolution_runs_reference="path/to/per_allele_deconv_directory",
    path_motif_selection_config="path/to/motif_selection_config.yml",
    path_contaminant_types_config="path/to/contaminant_types_config.yml",
    path_plot_results="path/to/output_directory/summary_plot.pdf",
)
```
