# Expectation-Maximization Algorithm

Fitting mixture models using expectation-maximization (EM) algorithms is a classical bioinformatics
approach to _de novo_ sequence motif discovery
[(Bailey and Elkan, 1994)](https://pubmed.ncbi.nlm.nih.gov/7584402/).

## EM-based deconvolution for MHC-II

EMMo implements a version of [MoDec](https://github.com/GfellerLab/MoDec)
[(Racle et al., 2019)](https://www.nature.com/articles/s41587-019-0289-6), which is an EM-based
method adapted for the purpose of (i) identification of MHC-II binding core motifs and (ii)
deconvolution of multi-allelic of MHC-II ligand samples. The method takes as input a list of $N$
peptides and (a range of) the number $K$ of classes, which correspond to the alleles/specificities
that are expected to be contained in the list. The EM algorithm then aims at maximizing the
following log likelihood function:

$$
\log(\mathcal{L}) = \sum_{n=1}^{N}
\left(
    W^n \cdot
    \log \left(
        \sum_{k=0}^{K} \sum_{s=-S}^{S} w_{k,s}
        \prod_{l=1}^{L}
        \frac{
            \theta_{l, x_{l \bigoplus s}^{n}}^{k}
        }{
            f_{x_{l \bigoplus s}^{n}}
        }
    \right)
\right)
+ \log(P(\theta))
$$

The parameters to be fitted are

- $\theta^k$, $1\le k\le K$, the $k$th binding core motif of length $L$
- $w_{k,s}$ the weight (or prior probability) of class $k$ and offset $s$

To account for the binding core offset preferences of MHC-II molecules towards the center of the
peptide, the offset weights $w_{k,s}$ to be fitted are shared between peptides of different lengths
and centered such that $s=0$ refers to the core that is exactly in the middle of the peptide. The
maximal shift $S$ from the center depends on the maximal peptide length in the sample and the
"special sum" $l \bigoplus s$ ensures the correct alignment of each peptide $x^n$, see
[(Racle et al., 2019, supplement)](https://www.nature.com/articles/s41587-019-0289-6). A fixed flat
motif $\theta_{l, i}^{0}$ models potential contaminant peptides and consist of the amino acid
frequencies in the human proteome. Similarly, the values $f_{i}^{n}$ are the fixed amino acid
frequencies observed in HLA-II ligands. Finally, $P(\theta)$ is a prior term and $W^n$ is a weight
assigned to the $n$th peptide intended to downweight peptides that share $9$-mers with other
peptides in the sample ($1/W^n$ is the average number of times each $9$-mer appears in the full
peptide list).

The EM algorithm computes so-called responsibilities during the expectation step in form of a matrix
of dimensions $(N,\; K +1,\; 2S +1)$. These can be interpreted as the probabilities that the $n$th
peptide belongs to class $k$ and its binding core offset is $s$. The final class and core prediction
is usually given by the maximal responsibility value per peptide when the EM algorithm has
converged.

Since the source code of MoDec is not available, some implementation details may differ in EMMo. For
example, we use a specific initialization for the responsibilities (rather than randomly
initializing $\theta$ and $w_{k, s}$) and then start with a maximization step. To be specific, we
randomly assign each peptide to a class (if possible, ensuring that at least one peptide is assigned
to each class) and use a uniform distribution for the offset dimension of the responsibility matrix
(taking only the possible offsets for the respective peptide length into account). The EM algorithm
then runs until the log likelihood difference between two steps falls below a user-defined threshold
(default at 1e-3).

Usually the EM algorithm is run multiple times with different initializations and the run with the
highest $\log(\mathcal{L})$ is reported as output.

We identified a bias affecting the offset weights $w_{k,s}$ that is introduced in MoDec due to the
manner in which the peptides of different lengths are aligned. With the "special sum"
$l \bigoplus s$, the offset $s=0$ is skipped for peptides for which the peptide and motif length
difference is odd (here the binding core cannot be exactly centered within the peptide). As a
consequence, these peptides do not contribute to the middle offset weights which are obtained in the
maximization step by summing up the responsibility matrix along the peptide dimension and
normalizing.

In EMMo, we fix this issue by upweighting the middle offset weight (before normalization) by a
factor that is equal to the total effective number of peptides divided by the effective number of
peptides that contribute to the middle offset weight.

<img src="../media/explanations/offset-preference-reverse-binders.png" width="800"/>

Panel A in the figure shows a the plot taken from
[Racle et al. (2023)](https://doi.org/10.1016/j.immuni.2023.03.009). The drop for offset $s=0$ is a
consequence of the above-described bias. The reproduction of the analysis using EMMo in panel B also
shows a slight shift of the distribution towards the N-terminus for the reverse binders, but without
the artificial drop at the middle offset. Similarly, another bias is introduced by the fact that
only the longer peptides contribute to offsets that are further away from zero. A possible future
experiment to fix this could be estimating a suitable offset weight distribution with the peptide
length and binding core offset as parameters instead of the current categorical distribution.
