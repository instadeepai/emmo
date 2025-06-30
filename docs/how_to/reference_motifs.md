# Reference motifs

## Background

Per-allele reference motifs are useful for many purposes including:

- Compiling a simple pMHC binding predictor based on position probability (PPMs) or position weight
  matrices (PWMs)
- Analyzing functional similarity of alleles (using motif similarity as a proxy)
- Identification of contaminants or MHC misannotations in mass spec data

A simple way to obtain per-allele reference motifs with EMMo is to apply the motif deconvolution to
single-allelic data, which is usually considered to have high quality. By setting the number of
motifs to be identified $K\ge 2$, potential contaminants can be further reduced resulting in more
robust motifs. However, it should be noted the some alleles have multiple subspecificities that
cannot be reflected well by a single motifs. The same is true for the reverse binder phenomenon that
has been observed for certain HLA-DP alleles.

## Usage in EMMo

The curation of reference motifs for alleles is still a manual process. Based on the output
directory of per-group (in this case per-allele) deconvolution, the motifs are selected using a YAML
config file. By default, for a given allele, the motif in the deconvolution run for $K=1$ is
selected. Selection of motifs from runs with $K\ge 2$ or whether to exclude an allele completely
must be specified in the config file. Reasons for such manual selections include the removal of the
contribution of contaminant peptides.

<details>
  <summary>Example config: Motif selection</summary>

The example is a manual curated collection for MSDB 2024-10 v1 (Fiigment MSDB), single-allelic MHC1
data, per-allele deconvolution $1\le K\le 3$. Additional available alleles are included with the
single motif in the run with $K=1$.

```yaml
A1101:
  keep: true
  classes: 2
  motif: 2
  comment: "remove C0401 contamination (C1R cell line)"
A2407:
  keep: true
  classes: 2
  motif: 2
  comment: "remove C0102 contamination (B721.221 cell line)"
A7401:
  keep: true
  classes: 2
  motif: 2
  comment: "remove C0102 contamination (B721.221 cell line)"
B1502:
  keep: true
  classes: 2
  motif: 1
  comment: "remove C0401 contamination (C1R cell line)"
C0102:
  keep: true
  classes: 2
  motif: 2
  comment: "remove C0401 contamination (C1R cell line)"
C0202:
  keep: true
  classes: 2
  motif: 1
  comment: "remove C0401 contamination (C1R cell line)"
C0303:
  keep: true
  classes: 2
  motif: 2
  comment: "remove C0401 contamination (C1R cell line)"
C0304:
  keep: true
  classes: 2
  motif: 1
  comment: "remove C0401 contamination (C1R cell line)"
C0401:
  keep: true
  classes: 2
  motif: 1
  comment: "remove B3503 contamination (C1R cell line)"
C0501:
  keep: true
  classes: 2
  motif: 1
  comment: "remove B3503 contamination (C1R cell line)"
C0602:
  keep: true
  classes: 2
  motif: 2
  comment: "remove C0401 contamination (C1R cell line)"
C0701:
  keep: true
  classes: 2
  motif: 1
  comment: "remove C0401 contamination (C1R cell line)"
C0702:
  keep: true
  classes: 2
  motif: 2
  comment: "remove C0401 contamination (C1R cell line)"
C0704:
  keep: true
  classes: 2
  motif: 1
  comment: "remove putative B3701 contamination"
C1203:
  keep: true
  classes: 2
  motif: 1
  comment: "remove C0401 contamination (C1R cell line)"
C1402:
  keep: true
  classes: 2
  motif: 2
  comment: "remove C0401 contamination (C1R cell line)"
C1502:
  keep: true
  classes: 2
  motif: 1
  comment: "remove C0401 contamination (C1R cell line)"
C1601:
  keep: true
  classes: 2
  motif: 1
  comment: "remove C0401 contamination (C1R cell line)"
C1701:
  keep: true
  classes: 2
  motif: 2
  comment: "remove C0401 contamination (C1R cell line)"
E0101:
  keep: false
  comment: "motif appears to be contaminant C0401 (C1R cell line)"
G0101:
  keep: true
  classes: 2
  motif: 1
  comment: "remove C0401 contamination (C1R cell line)"
```

</details>

The path to the per-allele deconvolution runs together with a YAML file defining the motif selection
is used as input for various functions in EMMo.

```python
from emmo.pipeline.model_selection import select_deconvolution_models

# Returns the selected models and motifs as a dictionary containing as keys the alleles
# (or groups) and as values dictionaries with keys 'classes', 'motif', and 'model_path'
selected_models = select_deconvolution_models(
    models_directory="path/to/per_allele_deconv_directory",
    selection_path="path/to/motif_selection_config.yml",
)
```

```python
from emmo.pipeline.model_selection import load_selected_ppms

# Returns a dictionary mapping the alleles to the selected motifs (PPM)
selected_models = load_selected_ppms(
    models_directory="path/to/per_allele_deconv_directory",
    selection_path="path/to/motif_selection_config.yml",
    mhc_class=1,
)
```
