"""Module for functions that parse the output of other tools."""
from __future__ import annotations

import numpy as np
from cloudpathlib import AnyPath

from emmo.constants import AA2IDX
from emmo.constants import NATURAL_AAS
from emmo.io.file import load_txt
from emmo.io.file import Openable
from emmo.resources.background_freqs import get_background


def get_class_weight_from_responsibilities_modec(
    directory: Openable, number_of_motifs: int
) -> np.ndarray:
    """Obtain class weights from MoDec output.

    Args:
        directory: Directory containing the MoDec output.
        number_of_motifs: The number of classes/motifs.

    Returns:
        The class weights.
    """
    directory = AnyPath(directory)

    n = number_of_motifs
    resp_file = directory / "Responsibilities" / f"fullPepRes_K{n}.txt"

    lines: list[str] = []

    for line in load_txt(resp_file):
        # header also starts with a valid character ('P')
        if line[0] in NATURAL_AAS:
            lines.append(line)
        else:
            lines[-1] += line

    print("Number of lines (incl.) header:", len(lines))

    header = lines[0].split("\t")
    print("Number of columns:", len(header))

    split_lines = [line.split("\t") for line in lines[1:]]
    for split_line in split_lines:
        if len(split_line) != len(header):
            raise ValueError(f"found line with {len(split_line)} values")

    max_offset = int(header[4][header[4].find(",") + 2 : header[4].find(")")])
    n_offsets = 2 * max_offset + 1

    responsibilities = np.zeros((len(split_lines), n + 1, n_offsets), dtype=np.float64)

    for s, split_line in enumerate(split_lines):
        # flat motif has index 0 in MoDec
        a = 4
        b = a + n_offsets
        for i, value in enumerate(split_line[a:b]):
            responsibilities[s, n, i] = float(value)

        for c in range(n):
            a = (c + 1) * n_offsets + 4
            b = a + n_offsets
            for i, value in enumerate(split_line[a:b]):
                responsibilities[s, c, i] = float(value)

    class_weights = np.sum(responsibilities, axis=0)
    class_weights /= np.sum(class_weights)

    return class_weights


def _parse_pwm_modec(file: Openable) -> np.ndarray:
    """Parse the matrices in the MoDec output.

    Args:
        file: The file to parse.

    Returns:
        The matrix.
    """
    pwm = np.zeros((9, 20), dtype=np.float64)

    lines = load_txt(file)

    for aa, line in enumerate(lines[2:]):
        split_line = line.split("\t")[1:]
        for k, value in enumerate(split_line):
            pwm[k, aa] = float(value)

    return pwm


def parse_pwms_modec(directory: Openable, number_of_motifs: int) -> np.ndarray:
    """Parse the MoDec output.

    Args:
        directory: Directory containing the MoDec output.
        number_of_motifs: The number of classes/motifs.

    Returns:
        The parsed matrices.
    """
    directory = AnyPath(directory)

    n = number_of_motifs
    pwms = np.zeros((n + 1, 9, 20), dtype=np.float64)

    pwms[n] = get_background(which="uniprot")

    for i in range(n):
        pwms[i] = _parse_pwm_modec(directory / "PWMs" / f"PWM_K{n}_{i+1}.txt")

    # normalize to make sure that frequencies sum to one for each position
    pwms /= np.sum(pwms, axis=2, keepdims=True)

    return pwms


def parse_sequences_modec(directory: Openable, number_of_motifs: int) -> list[str]:
    """Parse the sequences values from MoDec output.

    Args:
        directory: Directory containing the MoDec output.
        number_of_motifs: The number of classes/motifs.

    Returns:
        The parsed sequences.
    """
    directory = AnyPath(directory)
    file = directory / "Responsibilities" / f"bestPepResp_K{number_of_motifs}.txt"

    return [line.split("\t")[0] for line in load_txt(file)[1:]]


def loglikelihood_modec(sequences: list[str], pwms: np.ndarray, class_weights: np.ndarray) -> float:
    """Log likelihood given sequences and a model output by MoDec.

    Args:
        sequences: The sequences.
        pwms: The PWMs.
        class_weights: The class weights.

    Returns:
        The log likelihood.
    """
    n_sequences = len(sequences)
    n_classes, n_motif, _ = pwms.shape
    n_offsets = class_weights.shape[1]

    log_likelihood = 0.0

    p = np.zeros(n_sequences)

    for s, seq in enumerate(sequences):
        len_seq = len(seq)
        s_max = (len_seq - n_motif + 1) // 2
        k = (n_offsets - 1) // 2
        if (len_seq - n_motif) % 2 == 0:
            offset_list = [s + k for s in range(-s_max, s_max + 1)]
        else:
            offset_list = [s + k for s in range(-s_max, 0)]
            offset_list.extend([s + k for s in range(1, s_max + 1)])

        for c in range(n_classes):
            for i, o in enumerate(offset_list):
                prob = class_weights[c, o]
                for k in range(n_motif):
                    prob *= pwms[c, k, AA2IDX[seq[i + k]]]
                p[s] += prob

    log_likelihood = np.sum(np.log(p))

    # at the moment, we use pseudo_count_prior as a constant exponent of the Dirichlet priors,
    # hence we can do the following
    log_likelihood += 0.1 * np.sum(np.log(pwms[: n_classes - 1]))

    return log_likelihood


def _parse_pwm_mixmhcp(file_path: Openable) -> np.ndarray:
    """Parse a single PWM output of MixMCHp.

    Args:
        file: The file path.

    Returns:
        The parsed PWM.
    """
    pwm = np.zeros((9, 20), dtype=np.float64)

    lines = load_txt(file_path)

    for aa, line in enumerate(lines[1:]):
        split_line = line.split("\t")[1:]
        for k, value in enumerate(split_line):
            pwm[k, aa] = float(value)

    return pwm


def parse_pwms_mixmhcp(directory: Openable, number_of_motifs: int) -> np.ndarray:
    """Parse all PWMs output by MixMHCp.

    Args:
        directory: Directory containing the MixMHCp output.
        number_of_motifs: The number of classes/motifs.

    Returns:
        The parsed PWMs.
    """
    directory = AnyPath(directory)

    n = number_of_motifs
    pwms = np.zeros((n, 9, 20), dtype=np.float64)

    for i in range(n):
        pwms[i] = _parse_pwm_mixmhcp(directory / "Multiple_PWMs" / f"PWM_{n}_{i+1}.txt")

    # normalize to make sure that frequencies sum to one for each position
    pwms /= np.sum(pwms, axis=2, keepdims=True)

    return pwms


def parse_from_mhcmotifviewer(file_path: Openable, as_frequencies: bool = True) -> np.ndarray:
    """Obtain a position probability matrix from MHCMotifViewer files.

    Args:
        file_path: The file downloaded from MHCMotifViewer.
        as_frequencies: Whether to convert the scores to probabilities.

    Returns:
        The parsed PPM.
    """
    _, _, _, header, *lines = load_txt(file_path)
    amino_acids = header.split()
    sorted_index = {a: i for i, a in enumerate(sorted(amino_acids))}

    pwm = np.zeros((len(amino_acids), len(lines)))

    for j, line in enumerate(lines):
        for a, value in zip(amino_acids, line.split()[2:]):
            pwm[sorted_index[a], j] = float(value)

    if as_frequencies:
        # following is not ideal since we do not know how exactly the PWM was calculated for
        # MHCMotifViewer web service; however, we only want to obtain "some" frequency matrix that
        # we can work with
        b = get_background(which="uniprot")
        pwm = np.exp2(pwm) * b[:, np.newaxis]
        pwm /= np.sum(pwm, axis=0)

    return pwm
