# Identification of contaminants

The list of peptides obtained in a mass spec experiment typically contains both false positives (in
terms of the interpretation of the mass spectra) and contaminants (i.e., peptides that were
physically present in the mass spec experiment but not bound to the expected MHC molecules). The
former are addressed by setting a strict FDR threshold and considering additional metrics to improve
the peptide calling (predicted vs. observed HPLC retention time and/or fragment ion intensities).
The contaminants are partially addressed by using simple filters (lists of known contaminant
proteins, length ranges, tryptic peptides, etc.).

Remaining sources of contaminants include:

- [MHC1 ligands in MHC2 ligandomics data](./mhc1_in_mhc2_ligandomics_data.md) (and possibly vice
  versa)
- [peptides from unexpected alleles](./peptides_from_unexpected_alleles.md), mostly other alleles in
  single-allelic data that are not expected to be present (resulting from issues with the experiment
  such as incomplete removal of the cell line's original MHC molecules or spill over)
- [misannotation of the sample's alleles](./mhc_allele_misannotations.md) (swapping of MHC
  annotations, typos, etc.)
- peptides that were not presented by any MHC allele (missed contaminant proteins, etc.)
