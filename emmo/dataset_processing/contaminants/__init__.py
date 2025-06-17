"""Identification and annotation of potential contaminant peptides.

The list of peptides obtained in a mass spec experiment typically contains both false positives (in
terms of the interpretation of the mass spectra) and contaminants (i.e., peptides that were
physically present in the mass spec experiment but not bound to the expected MHC molecules). The
former are addressed by setting a strict FDR threshold and considering additional metrics to improve
the peptide calling (predicted vs. observed HPLC retention time and/or fragment ion intensities).
The contaminants are partially addressed using simple filters (lists of known contaminant proteins,
length range, tryptic peptides).

Remaining sources of contaminants include:
  - MHC1 ligands in MHC2 ligandomics data
  - peptides from other alleles in single-allelic data
  - misannotation of the sample's alleles
  - peptides that were not presented (missed contaminant proteins, …)

This module provides functions to identify such kinds of contaminants and to annotate them.
"""
from __future__ import annotations

from emmo.dataset_processing.contaminants.mhc1_in_mhc2 import (  # noqa
    annotate_mhc1_contaminants_in_mhc2_ligandomics,
)
from emmo.dataset_processing.contaminants.unexpected_alleles import (  # noqa
    annotate_peptides_from_unexpected_alleles,
)
