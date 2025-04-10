# EM Algorithm with Group-Specific Background

## MHC-I

EMMo implements a variant of the MHC-I deconvolution algorithm, that uses group-specific background
frequencies. A possible application of this variant is the removal of the peptide-MHC binding signal
in order to deconvolute other signals, e.g., from TCR binding in data from multimer assays. In such
a scenario, a position probability matrix (PPM) could be provided for each group (=allele) which is
then used as background during the EM algorithm.

### Log Likelihood Function

Like the standard MHC-I deconvolution, the method takes as input a list of $N$ peptides (**currently
only 9-mers are supported**) and (a range of) the number $K$ of classes, which correspond to the
motifs/specificities that are expected to be contained in the list. Additionally, a PPM $b$
containing the background frequencies must be provided for each group.

The EM algorithm then aims to maximize the following log likelihood function:

$$
\log(\mathcal{L}) = \sum_{n=1}^{N}
\log \left(
    \sum_{k=0}^{K}
    w^k
    \prod_{l=1}^{9}
    \frac{
        \theta_{l, x_l^n}^{k}
    }{
        b_{l, x_{l}^{n}}^n
    }
\right) +
\log(P(\theta))
$$

where $P(\theta)$ is a prior term, $b^n$ is the background PPM of the group of peptide $n$, and
$b_{l, x_{l}^{n}}^n$ is an entry of the background PPM at motif position $l$ and amino acid
$x_{l}^{n}$.

The parameters to be fitted are:

- $\theta^k$, $1\le k\le K$, the $k$-th motif
- $w^k$ the weight (or prior probability) of class $k$

Note that the $\theta_{l, i}^{0}$ are fixed. They model potential contaminant peptides and consist
of the amino acid frequencies in the human proteome (for a given residue $i$, the frequencies are
identical for all $l$).

### From PPMs to PSSMs

In the maximization steps of the EM algorithm, the PPMs $\theta^k$ are estimated using the peptide
sequences and their responsibilities towards the individual classes, without any correction based on
the per-group background frequencies. As a consequence, the signal that is intended to be removed is
still contained in the PPMs $\theta^k$.

We therefore aim to construct position-specific scoring matrices (PSSMs) in which the background
signal is removed as good as possible. To this end, an average of the background PPMs that is
weighted by the responsibilities is additionally computed at the end of the deconvolution. Let $A$
be the alphabet (i.e., the set of amino acids) and let $\rho_{k}^{n}$ be the responsibility of
peptide $n$ towards class $k$. Then, the background PPM $\beta^{k}$ of class $k$ is given by

$$
\beta_{l, a}^{k} = \frac{
    \sum_{n=1}^{N} \rho_{k}^{n} \cdot b_{l, a}^n
}{
    \sum_{a'\in A} \sum_{n=1}^{N} \rho_{k}^{n} \cdot b_{l, a'}^n
}
$$

The denominator normalizes the frequencies such that they again sum up to 1 for each position in
each background PPM.

Finally, the PSSMs $S^{k}$ are given by

$$
S_{l, a}^{k} = \log_2 \left(
    \frac{
        \theta_{l, a}^{k}
    }{
        \beta_{l, a}^{k}
    }
\right)
$$

### Usage in EMMo

The above-described variant of the deconvolution algorithm is implemented in
`emmo.em.experimental_modifications.mhc1_per_group_background`. The following code snippet is an
example of how to use it:

```python
from emmo.pipeline.sequences import SequenceManager
from emmo.em.experimental_modifications.mhc1_per_group_background import (
    EMRunnerMHC1PerGroupBackground,
)
# input CSV file with columns "peptide" and "allele"
# (must only contain peptides of length MOTIF_LENGTH)
INPUT_FILE = "peptides.csv"

OUTPUT_DIR = "output"
MOTIF_LENGTH = 9
NUMBER_OF_CLASSES = 3   # number of motifs to deconvolute
NUMBER_OF_RUNS = 20     # number of runs (with different initializations)

sequence_manager = SequenceManager.load_from_csv(
    INPUT_FILE,
    sequence_column="peptide",
    additional_columns=["allele"],
)

# add code to load/generate 'group2ppm' (the per-group background PPMs),
# this needs to be a dictionary mapping the groups (usually str) to numpy
# arrays of dimensions (MOTIF_LENGTH, len(sequence_manager.alphabet))

em_runner = EMRunnerMHC1PerGroupBackground(
    sequence_manager,
    MOTIF_LENGTH,
    NUMBER_OF_CLASSES,
    group2ppm,
    group_attribute="allele",
)

em_runner.run(OUTPUT_DIR, n_runs=NUMBER_OF_RUNS)
```

The background PPMs for the individual classes are saved as part of the model artifacts. Together
with the main PPMs of the deconvolution result, they can be used to compute the above-described
PSSMs:

```python
import numpy as np
from emmo.models.deconvolution import DeconvolutionModelMHC1

model = DeconvolutionModelMHC1.load(OUTPUT_DIR)
pssm = np.log2(model.ppm / model.artifacts["background_ppm"])
```
