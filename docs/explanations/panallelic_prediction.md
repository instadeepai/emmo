# Pan-allelic Prediction

Since the EMMo predictors use PPMs that have been obtained for the alleles seen in the deconvolution
step (i.e., the "training"), they are not pan-allelic predictors by design (similar to the original
version of MixMHC2pred).

To be able to score also peptides with associated alleles that have not been seen in the training
(i.e., for which no PPMs are available), we use the PPM of the nearest neighbour among the available
alleles. To determine the nearest neighbour, we use the following distance function from
[Nielsen et al. (2007)](http://dx.doi.org/10.1371/journal.pone.0000796):

$$
d(A,B) = 1  - \frac{s(A,B)}{\sqrt{s(A, A)\cdot s(B, B)}}
$$

Here, $A$ and $B$ are the pseudosequences of two alleles (where $\alpha$ and $\beta$ chain are
concatenated in case of MHC-II) and $s(S_1, S_2)$ is the BLOSUM62 score for two aligned amino acid
sequences $S_1$ and $S_2$.

MixMHC2pred 2.0 [Racle et al. (2023)](https://doi.org/10.1016/j.immuni.2023.03.009) uses an
alternative strategy to obtain a pan-allelic predictor from a limited set of available motifs. The
method uses two neural networks where the first one predicts a PPM from the pseudosequence of the
allele. This PPM is then used to calculate an intermediate binding score, which then serves as input
for the second neural network (among other features). We plan to also explore this approach in the
future.
