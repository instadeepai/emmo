"""Module containing the length distributions."""
from __future__ import annotations


# from BioNDeep MHC class I data (merge train and tune partition, pseudocount of 1 added for each
# length in [min, max] range)
# (gs://biondeep-data/datasets/mhc1/binding/MSDF_20200604/)
LENGTH_DISTRIBUTION_MS_MHC1_DATA = {
    8: 0.09634585,
    9: 0.62028448,
    10: 0.15641970,
    11: 0.09439488,
    12: 0.03255509,
}

# from BioNDeep MHC class II data (merge train and tune partition, pseudocount of 1 added for each
# length in [min, max] range)
# (gs://biondeep-data/datasets/mhc2/binding/)
LENGTH_DISTRIBUTION_MS_MHC2_DATA = {
    9: 0.01919552,
    10: 0.01951503,
    11: 0.02693931,
    12: 0.05061631,
    13: 0.07996872,
    14: 0.11391192,
    15: 0.13506651,
    16: 0.13939664,
    17: 0.11913330,
    18: 0.08313014,
    19: 0.05062472,
    20: 0.03335463,
    21: 0.02319774,
    22: 0.01604251,
    23: 0.01489061,
    24: 0.01145173,
    25: 0.00966082,
    26: 0.00795399,
    27: 0.00692821,
    28: 0.00645736,
    29: 0.00476735,
    30: 0.00468327,
    31: 0.00434695,
    32: 0.00322027,
    33: 0.00373316,
    34: 0.00339684,
    35: 0.00301007,
    36: 0.00235425,
    37: 0.00123598,
    38: 0.00068105,
    39: 0.00042881,
    40: 0.00015134,
    41: 0.00011771,
    42: 0.00016816,
    43: 0.00016816,
    44: 0.00001682,
    45: 0.00000841,
    46: 0.00001682,
    47: 0.00003363,
    48: 0.00002522,
}


def get_length_distribution(which: str) -> dict[int, float]:
    """Get the length distribution.

    Args:
        which: The name of the background distribution.

    Raises:
        ValueError: If the specified name is not available.

    Returns:
        The length distribution.
    """
    if which == "MHC1_biondeep":
        return LENGTH_DISTRIBUTION_MS_MHC1_DATA
    elif which == "MHC2_biondeep":
        return LENGTH_DISTRIBUTION_MS_MHC2_DATA
    else:
        raise ValueError(f"background '{which}' is not available")


if __name__ == "__main__":
    print(get_length_distribution(which="MHC1_biondeep"))
    print(get_length_distribution(which="MHC2_biondeep"))
