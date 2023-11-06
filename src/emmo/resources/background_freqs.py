"""Module containing the background frequencies."""
import numpy as np

# from https://github.com/GfellerLab/MixMHCp, needs to be re-checked
BACKGROUND_UNIPROT = {
    "A": 0.0702,
    "C": 0.0230,
    "D": 0.0473,
    "E": 0.0710,
    "F": 0.0365,
    "G": 0.0657,
    "H": 0.0263,
    "I": 0.0433,
    "K": 0.0572,
    "L": 0.0996,
    "M": 0.0213,
    "N": 0.0359,
    "P": 0.0631,
    "Q": 0.0477,
    "R": 0.0564,
    "S": 0.0833,
    "T": 0.0536,
    "V": 0.0597,
    "W": 0.0122,
    "Y": 0.0267,
}

# from BioNDeep MHC class I data (merge train, test, and tune partition)
# (gs://biondeep-data/datasets/mhc1/binding/MSDF_20200604/)
BACKGROUND_MS_MHC1_DATA = {
    "A": 0.076584,
    "C": 0.005569,
    "D": 0.044324,
    "E": 0.063805,
    "F": 0.055308,
    "G": 0.043486,
    "H": 0.030039,
    "I": 0.060332,
    "K": 0.049878,
    "L": 0.114874,
    "M": 0.018129,
    "N": 0.029773,
    "P": 0.061633,
    "Q": 0.038834,
    "R": 0.051136,
    "S": 0.067713,
    "T": 0.053714,
    "V": 0.080919,
    "W": 0.010045,
    "Y": 0.043906,
}

# from BioNDeep MHC class II data (merge train, test, and tune partition)
# (gs://biondeep-data/datasets/mhc2/binding/)
BACKGROUND_MS_MHC2_DATA = {
    "A": 0.086865,
    "C": 0.004767,
    "D": 0.049174,
    "E": 0.070215,
    "F": 0.031256,
    "G": 0.067076,
    "H": 0.025248,
    "I": 0.051752,
    "K": 0.066980,
    "L": 0.086119,
    "M": 0.012101,
    "N": 0.038058,
    "P": 0.072285,
    "Q": 0.048338,
    "R": 0.058607,
    "S": 0.076822,
    "T": 0.052485,
    "V": 0.070511,
    "W": 0.006022,
    "Y": 0.025322,
}

# amino acid frequencies (amino acids alphabetically sorted) as used by PSI-BLAST, i.e., taken from
# Robinson, A.B. & Robinson, L.R. (1991) "Distribution of glutamine and asparagine residues and
# their near neighbors in peptides and proteins." Proc. Natl. Acad. Sci. USA 88:8880-8884.
BACKGROUND_PSI_BLAST = {
    "A": 0.078047,
    "C": 0.019246,
    "D": 0.053640,
    "E": 0.062949,
    "F": 0.038556,
    "G": 0.073772,
    "H": 0.021992,
    "I": 0.051420,
    "K": 0.057438,
    "L": 0.090191,
    "M": 0.022425,
    "N": 0.044873,
    "P": 0.052028,
    "Q": 0.042644,
    "R": 0.051295,
    "S": 0.071198,
    "T": 0.058413,
    "V": 0.064409,
    "W": 0.013298,
    "Y": 0.032165,
}


def get_background(which: str = "uniprot") -> np.ndarray:
    """Get the background distribution.

    Args:
        which: The name of the background distribution.

    Raises:
        ValueError: If the specified name is not available.

    Returns:
        The background distribution.
    """
    if which == "uniprot":
        freqs = BACKGROUND_UNIPROT
    elif which == "MHC1_biondeep":
        freqs = BACKGROUND_MS_MHC1_DATA
    elif which == "MHC2_biondeep":
        freqs = BACKGROUND_MS_MHC2_DATA
    elif which == "psi_blast":
        freqs = BACKGROUND_PSI_BLAST
    else:
        raise ValueError(f"background '{which}' is not available")

    background = np.asarray([freqs[a] for a in sorted(freqs.keys())])

    # make sure frequencies sum to 1
    background /= np.sum(background)

    return background


if __name__ == "__main__":
    print(get_background(which="uniprot"))
    print(get_background(which="MHC2_biondeep"))
