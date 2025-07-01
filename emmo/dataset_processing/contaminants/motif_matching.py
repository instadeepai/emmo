"""Function for matching deconvoluted motifs to reference motifs.

This module provides functionality to match motifs (PPMs) from deconvolution runs to reference
motifs. This is for example useful to identify mass spec experiments with MHC allele misannotations
(e.g., swapped alleles or typos) or low-quality experiments (as indicated by poor motif quality).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from emmo.io.file import load_csv
from emmo.io.file import Openable
from emmo.io.output import find_deconvolution_results
from emmo.models.deconvolution import DeconvolutionModel
from emmo.models.deconvolution import DeconvolutionModelMHC1
from emmo.models.deconvolution import DeconvolutionModelMHC2
from emmo.pipeline.model_selection import load_selected_ppms
from emmo.utils import logger
from emmo.utils.alleles import split_and_shorten_alleles
from emmo.utils.sequence_distance import nearest_neighbors
from emmo.utils.statistics import jensen_shannon_divergence

log = logger.get(__name__)


def match_motifs_to_reference(
    path_deconvolution_runs: Openable,
    path_metadata_table: Openable,
    path_per_allele_reference: Openable,
    path_per_allele_reference_selection: Openable,
    mhc_class: int,
    number_of_classes: int,
    metadata_group_column: str = "experiment_id",
    metadata_alleles_column: str = "allele",
) -> pd.DataFrame:
    """Match motifs from deconvolution runs to reference motifs.

    This function loads deconvolution results, merges them with metadata, and matches the motifs
    (PPMs) to reference motifs. It returns a DataFrame with the best matching motifs for each motif
    in each deconvolution run. The DataFrame has at least the following columns:
    - `group`: Group identifier for the deconvolution run (e.g., the experiment ID).
    - `number_of_classes`: Number of classes that were considered.
    - `model_path`: Path to the deconvolution model.
    - `alleles`: Alleles from column `metadata_alleles_column` in the metadata table.
    - `class`: Class/cluster index (corresponding to one PPM), 1-based.
    - `best_matching_allele_genotype`: Best matching allele in the provided set of alleles (i.e.,
        in the genotype).
    - `best_matching_distance_genotype`: Distance to the best matching allele in the provided set
        of alleles.
    - `best_matching_allele_is_in_reference_genotype`: Whether the best matching allele in the
        genotype is in the set of reference alleles.
    - `best_matching_allele_overall`: Best matching allele across all reference alleles.
    - `best_matching_distance_overall`: Distance to the overall best matching allele.
    - `best_matching_allele_is_in_reference_overall`: Whether the overall best matching allele is
        in the set of reference alleles (should always be True by construction).

    Args:
        path_deconvolution_runs: Path to the directory containing deconvolution runs.
        path_metadata_table: Path to the metadata table CSV file.
        path_per_allele_reference: Path to the deconvolution runs containing the per-allele
            reference PPMs.
        path_per_allele_reference_selection: Path to the per-allele reference selection YAML file.
        mhc_class: MHC class (1 or 2).
        number_of_classes: Number of classes to consider. Used to filter the deconvolution runs.
        metadata_group_column: Column name in the metadata table that contains group information.
            Used to merge metadata with deconvolution results.
        metadata_alleles_column: Column name in the metadata table that contains allele information.

    Returns:
        A DataFrame with the best matching motifs for each deconvolution run, including details like
        the best matching allele, distance, and whether the allele is in the reference set.

    Raises:
        ValueError: If `mhc_class` is not 1 or 2, or if no deconvolution results are found for the
            specified number of classes.
    """
    if mhc_class not in (1, 2):
        raise ValueError("'mhc_class' must be either 1 or 2")

    allele2ppm = load_selected_ppms(
        models_directory=path_per_allele_reference,
        selection_path=path_per_allele_reference_selection,
        mhc_class=mhc_class,
    )

    df = find_deconvolution_results(path_deconvolution_runs)

    df = df[df["number_of_classes"] == number_of_classes].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"No deconvolution results found for {number_of_classes} classes in "
            f"{path_deconvolution_runs}"
        )

    df = _merge_metadata(
        df,
        path_metadata_table,
        metadata_group_column,
        metadata_alleles_column,
    )
    log.info("Merged deconvolution results with metadata.")

    query_alleles = df["alleles"].apply(split_and_shorten_alleles).explode().unique()
    query_alleles.sort()

    allele2reference_allele = map_alleles_to_reference_alleles(
        query_alleles=query_alleles.tolist(),
        reference_alleles=list(allele2ppm.keys()),
        mhc_class=mhc_class,
    )
    log.info(f"Mapped {len(allele2reference_allele):,} alleles to their nearest reference alleles.")

    log.info("Assigning best matching motifs for each deconvolution run ...")
    df["best_matching_motifs"] = df.apply(
        lambda row: _assign_best_matching_motifs(
            model_path=row["model_path"],
            alleles=row["alleles"],
            mhc_class=mhc_class,
            allele2ppm=allele2ppm,
            allele2reference_allele=allele2reference_allele,
        ),
        axis=1,
    )
    log.info("Finished assigning best matching motifs for each deconvolution run.")

    # expand the dictionaries in column "best_matching_motifs" into separate rows and columns
    df = df.explode("best_matching_motifs", ignore_index=True)
    df = pd.concat(
        [df.drop(columns=["best_matching_motifs"]), pd.json_normalize(df["best_matching_motifs"])],
        axis=1,
    )

    return df


def map_alleles_to_reference_alleles(
    query_alleles: list[str],
    reference_alleles: list[str],
    mhc_class: int,
) -> dict[str, tuple[list[str], float]]:
    """Map query alleles to nearest reference alleles.

    This function finds the reference alleles for a list of query alleles. It uses the
    `nearest_neighbors` function to find the closest available alleles whenever a query allele is
    not present in the set of reference alleles.

    Args:
        query_alleles: List of alleles to be mapped.
        reference_alleles: List of available reference alleles.
        mhc_class: MHC class (1 or 2).

    Returns:
        A dictionary mapping each query allele to a tuple containing the nearest reference alleles
        (if available, this is the allele itself) and the distance to them.

    Raises:
        ValueError: If `mhc_class` is not 1 or 2.
    """
    if mhc_class not in (1, 2):
        raise ValueError("'mhc_class' must be either 1 or 2")

    unavailable_alleles = set(query_alleles) - set(reference_alleles)

    if unavailable_alleles:
        log.info(
            f"Finding nearest reference alleles for {len(unavailable_alleles):,} (out of"
            f" {len(query_alleles):,}) that are missing in the reference"
        )

    unavailable_allele2nearest_reference_allele = nearest_neighbors(
        available_alleles=reference_alleles,
        unavailable_alleles=list(unavailable_alleles),
        mhc_class=1,
    )

    full_mapping = {
        allele: unavailable_allele2nearest_reference_allele.get(allele, ([allele], 0.0))
        for allele in query_alleles
    }

    return full_mapping


def _merge_metadata(
    df: pd.DataFrame,
    path_metadata_table: Openable,
    metadata_group_column: str,
    metadata_alleles_column: str,
) -> pd.DataFrame:
    """Merge deconvolution results with metadata.

    Args:
        df: DataFrame containing deconvolution results.
        path_metadata_table: Path to the metadata table CSV file.
        metadata_group_column: Column name in the metadata table that contains group information.
            Used to merge metadata with the deconvolution results.
        metadata_alleles_column: Column name in the metadata table that contains allele information.

    Returns:
        A DataFrame with the deconvolution results merged with the metadata table.

    Raises:
        ValueError: If the metadata table does not contain the required columns, if there are
            missing groups, or if there are null values in the alleles column.
    """
    df_metadata = load_csv(path_metadata_table)
    if not {metadata_group_column, metadata_alleles_column}.issubset(df_metadata.columns):
        raise ValueError(
            f"Metadata table must contain columns '{metadata_group_column}' and "
            f"'{metadata_alleles_column}'"
        )

    # normalize the group and alleles columns
    df_metadata["group"] = df_metadata[metadata_group_column].astype(str)
    df_metadata["alleles"] = df_metadata[metadata_alleles_column].astype(str)

    if not set(df["group"].unique()).issubset(df_metadata["group"].unique()):
        raise ValueError("deconvolution results contain groups not present in the metadata table")

    df_merged = df.merge(df_metadata, on="group", how="left", validate="many_to_one")

    if df_merged["alleles"].isnull().any():
        raise ValueError(f"metadata table column '{metadata_alleles_column}' contains null values")

    return df_merged


def _assign_best_matching_motifs(
    model_path: Openable,
    alleles: str,
    mhc_class: int,
    allele2ppm: dict[str, np.ndarray],
    allele2reference_allele: dict[str, tuple[list[str], float]],
) -> list[dict[str, Any]]:
    """Assign the best matching motifs for a given deconvolution model.

    This function computes the Jensen-Shannon divergence between the PPMs of the deconvolution
    model and the reference PPMs. It returns a list of dictionaries containing the best matching
    alleles and their distances for each class/cluster in the deconvolution model.
    Two types of best matching alleles are returned:
    1. The best matching allele across all reference alleles ('overall')
    2. The best matching allele in the provided set of alleles (i.e., the 'genotype').

    Args:
        model_path: Path to the deconvolution model.
        alleles: Alleles from the metadata table.
        mhc_class: MHC class (1 or 2).
        allele2ppm: Dictionary mapping reference alleles to their corresponding PPMs.
        allele2reference_allele: Dictionary mapping alleles to their nearest reference alleles and
            distances.

    Returns:
        A list of dictionaries with the best matching motifs for each class/cluster.
    """
    result: list[dict[str, Any]] = []

    # load the models
    model: DeconvolutionModel
    if mhc_class == 1:
        model = DeconvolutionModelMHC1.load(model_path)
    else:
        model = DeconvolutionModelMHC2.load(model_path)

    for c in range(model.number_of_classes):
        ppm = model.ppm[c]

        kl_distances = {
            ref_allele: jensen_shannon_divergence(ppm, ref_ppm)
            for ref_allele, ref_ppm in allele2ppm.items()
        }

        # overall best matching allele
        best_matching_allele_overall = min(kl_distances, key=kl_distances.get)
        best_matching_distance_overall = kl_distances[best_matching_allele_overall]

        # best matching allele in the provided set of alleles (i.e., the genotype)
        best_matching_allele_genotype = None
        best_matching_distance_genotype = float("inf")

        for allele in split_and_shorten_alleles(alleles):
            # there can be multiple reference alleles for a given allele (if they have the same
            # distance)
            ref_alleles = allele2reference_allele[allele][0]

            # take the mean distance over all available reference alleles
            distance = np.mean([kl_distances[ref_allele] for ref_allele in ref_alleles])
            if distance < best_matching_distance_genotype:
                best_matching_allele_genotype = allele
                best_matching_distance_genotype = distance

        result.append(
            {
                "class": c + 1,
                "best_matching_allele_overall": best_matching_allele_overall,
                "best_matching_distance_overall": best_matching_distance_overall,
                "best_matching_allele_is_in_reference_overall": best_matching_allele_overall
                in allele2ppm,
                "best_matching_allele_genotype": best_matching_allele_genotype,
                "best_matching_distance_genotype": best_matching_distance_genotype,
                "best_matching_allele_is_in_reference_genotype": best_matching_allele_genotype
                in allele2ppm,
            }
        )

    return result
