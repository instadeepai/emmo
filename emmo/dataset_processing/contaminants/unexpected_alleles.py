"""Identification and annotation of ligands from unexpected alleles."""
from __future__ import annotations

from typing import Any

import pandas as pd
from cloudpathlib import AnyPath

from emmo.io.file import load_csv
from emmo.io.file import load_yml
from emmo.io.file import Openable
from emmo.io.output import find_deconvolution_results
from emmo.models.deconvolution import DeconvolutionModelMHC1
from emmo.pipeline.model_selection import select_deconvolution_models
from emmo.utils import logger
from emmo.utils.viz.contaminants import plot_contaminant_counts

log = logger.get(__name__)

# Number of classes to use from the per-experiment deconvolution runs
NUMBER_OF_CLASSES_TO_USE = 1

# Columns to use from the experiment table
EXPERIMENT_COLUMNS = [
    "experiment_id",
    "experiment_type",
    "mhc_class",
    "allele",
    "cell_type",
    "dataset_name",
    "sample_alias",
    "antibody_or_treatment",
]

CONTAMINANT_SPEC_REQUIRED_KEYS = ["identifier", "column", "value", "contaminant_allele"]
CONTAMINANT_SPEC_REQUIRED_KEYS_EXCLUDE = ["column", "value"]
CONTAMINANT_SPEC_MATCH_MODES = ["equals", "equals_insensitive", "contains", "contains_insensitive"]


def annotate_peptides_from_unexpected_alleles(
    path_deconvolution_runs: Openable,
    path_experiments_table: Openable,
    path_deconvolution_runs_reference: Openable,
    path_motif_selection_config: Openable,
    path_contaminant_types_config: Openable,
    keep_all_responsibility_columns: bool = False,
    path_plot_results: Openable | None = None,
    num_experiments_per_plot: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annotate peptides from unexpected alleles with contaminant information.

    Peptides from the responsibility files are annotated with contaminant information based on
    (1) the provided contaminant specification in YAML format and (2) per-allele reference models.
    A peptide is annotated as True for a specific contaminant type if the score of the peptide
    given by the model from the same deconvolution run is lower than the score of the peptide
    given by the reference model for that contaminant type (i.e., for the associated allele from the
    contaminant specification).

    Args:
        path_deconvolution_runs: Path to the directory containing per-experiment deconvolution runs
            from which to annotate the peptides.
        path_experiments_table: Path to the experiment table in CSV format.
        path_deconvolution_runs_reference: Path to the directory containing per-allele reference
            deconvolution runs.
        path_motif_selection_config: Path to the motif selection file for the reference models
            (in YAML format).
        path_contaminant_types_config: Path to the contaminant specification file in YAML format.
        keep_all_responsibility_columns: If True, keep all columns from the responsibility files in
            the output.
        path_plot_results: Path to save the plot results. If None, no plot is saved.
        num_experiments_per_plot: Number of experiments to plot per subplot.

    Returns:
        Tuple of two DataFrames:
            (1) DataFrame containing the deconvolution models and their corresponding
                experiment information.
            (2) DataFrame containing the annotated peptides with contaminant information.
    """
    allele2reference_model = _load_reference_models_per_allele(
        path_deconvolution_runs_reference,
        path_motif_selection_config,
    )
    log.info(f"Loaded reference models for {len(allele2reference_model):,} alleles.")

    df_models = find_deconvolution_results(path_deconvolution_runs)
    df_models = df_models[df_models["number_of_classes"] == NUMBER_OF_CLASSES_TO_USE].reset_index(
        drop=True
    )
    df_models["experiment_id"] = df_models["group"].apply(_infer_experiment_id)
    df_experiments = load_csv(path_experiments_table, usecols=EXPERIMENT_COLUMNS)
    df_models = df_models.merge(
        df_experiments,
        on="experiment_id",
        how="left",
        validate="1:1",
    )
    log.info(f"Loaded table with {len(df_models):,} experiments.")

    contaminants = _load_and_check_contaminant_specification(path_contaminant_types_config)
    log.info(f"Loaded contaminant specification with {len(contaminants):,} entries.")

    contaminant_columns = []
    for contaminant in contaminants:
        column = f"contaminant_{contaminant['identifier']}"
        contaminant_columns.append(column)

        df_models[f"contaminant_{contaminant['identifier']}"] = df_models.apply(
            _check_row_contaminant_match,
            contaminant=contaminant,
            axis=1,
        )

    df_peptides = pd.concat(
        [
            _annotate_single_experiments(row, contaminants, allele2reference_model)
            for _, row in df_models.iterrows()
        ],
        ignore_index=True,
    )

    df_peptides["contaminant_any"] = df_peptides[contaminant_columns].max(axis=1).astype(bool)

    _log_contaminant_counts(df_peptides, contaminants)

    if path_plot_results is not None:
        log.info("Plotting contaminant counts ...")
        plot_contaminant_counts(
            df_peptides,
            contaminants,
            sort_by_total_contaminants=True,
            num_experiments_per_plot=num_experiments_per_plot,
            save_as=path_plot_results,
        )

    if not keep_all_responsibility_columns:
        df_peptides = df_peptides[
            ["peptide", "group", "experiment_id"] + contaminant_columns + ["contaminant_any"]
        ]

    return df_models, df_peptides


def _load_reference_models_per_allele(
    path_deconvolution_runs: Openable,
    path_model_selection: Openable,
) -> dict[str, DeconvolutionModelMHC1]:
    """Load reference models for each allele.

    Args:
        path_deconvolution_runs: Path to the directory containing deconvolution runs.
        path_model_selection: Path to the model selection file.

    Returns:
        Dictionary mapping alleles to their corresponding deconvolution models.
    """
    selected_models = select_deconvolution_models(path_deconvolution_runs, path_model_selection)

    allele2reference_model = {}
    for allele, selected_model in selected_models.items():
        model = DeconvolutionModelMHC1.load(selected_model["model_path"])
        ppm = model.ppm[selected_model["motif"] - 1]
        allele2reference_model[allele] = DeconvolutionModelMHC1.build_from_ppm(ppm)

    return allele2reference_model


def _infer_experiment_id(group: str) -> str:
    """Extract the experiment ID from the group name.

    Args:
        group: The group string containing the experiment ID (starting with "exp").

    Returns:
        The extracted experiment ID.

    Raises:
        ValueError: If the experiment ID is not found in the group.
    """
    for item in group.split("_"):
        if item.startswith("exp"):
            return item

    raise ValueError(f"Experiment ID not found in group: {group}")


def _load_and_check_contaminant_specification(path: Openable) -> list[dict[str, Any]]:
    """Load and check the contaminant specification from a YAML file.

    This function loads the contaminant specification from the given path and checks if it is
    valid. The specification must be a list of dictionaries, each containing the keys 'identifier',
    'column', 'value', and 'contaminant_allele'. The optional key 'value_match' can be used to
    specify the matching mode for the 'value' key. The matching modes can be 'equals',
    'equals_insensitive', 'contains', 'contains_insensitive'. Additionally, each contaminant
    specification may contain an optional 'exclude' key, which must be a dictionary containing the
    keys 'column' and 'value'. The 'exclude' key is used to specify conditions under which the
    contaminant check should be ignored.

    Args:
        path: Path to the contaminant specification file.

    Returns:
        List of contaminant specifications.

    Raises:
        ValueError: If the contaminant specification is invalid.
    """
    contaminants = load_yml(path).get("contaminant_alleles", [])

    if not isinstance(contaminants, list):
        raise ValueError("Contaminant specification must be a list.")

    for contaminant in contaminants:
        _check_contaminant_specification(contaminant)

    return contaminants


def _check_contaminant_specification(contaminant: dict[str, Any]) -> None:
    """Check if the contaminant specification is valid.

    Args:
        contaminant: Dictionary containing the contaminant specification.

    Raises:
        ValueError: If the contaminant specification is invalid.
    """
    if not isinstance(contaminant, dict):
        raise ValueError("Contaminant specification must be a dictionary.")

    missing_keys = sorted(set(CONTAMINANT_SPEC_REQUIRED_KEYS) - set(contaminant))
    if missing_keys:
        raise ValueError(f"Contaminant specification must have the following keys: {missing_keys}.")

    if contaminant.get("value_match", "equals") not in CONTAMINANT_SPEC_MATCH_MODES:
        raise ValueError(f"'value_match' must be one of {CONTAMINANT_SPEC_MATCH_MODES}.")

    if "exclude" not in contaminant:
        return

    if not isinstance(contaminant["exclude"], dict):
        raise ValueError("Contaminant specification 'exclude' must be a dictionary.")

    missing_keys = sorted(set(CONTAMINANT_SPEC_REQUIRED_KEYS_EXCLUDE) - set(contaminant["exclude"]))
    if missing_keys:
        raise ValueError(
            f"Contaminant specification 'exclude' must have the following keys: {missing_keys}."
        )

    if contaminant["exclude"].get("value_match", "equals") not in CONTAMINANT_SPEC_MATCH_MODES:
        raise ValueError(f"'value_match' must be one of {CONTAMINANT_SPEC_MATCH_MODES}.")


def _check_row_match(row: pd.Series, criteria_dict: dict[str, Any]) -> bool:
    """Check if the row matches the criteria.

    Args:
        row: The row to check.
        criteria_dict: Dictionary containing the criteria for matching. It must have the keys
            'column' and 'value'. Additionally, it can have the key 'value_match' to specify the
            matching mode ('equals', 'equals_insensitive', 'contains', 'contains_insensitive').
            The default is 'equals'.

    Returns:
        True if the row matches the criteria, False otherwise.
    """
    match_mode = criteria_dict.get("value_match", "equals")
    query = criteria_dict["value"]
    target = row[criteria_dict["column"]]

    if match_mode == "equals":
        return target == query
    elif match_mode == "equals_insensitive":
        return target.lower() == query.lower()
    elif match_mode == "contains":
        return query in target
    elif match_mode == "contains_insensitive":
        return query.lower() in target.lower()
    else:
        raise ValueError(f"Unknown match mode: {match_mode}")


def _check_row_contaminant_match(row: pd.Series, contaminant: dict[str, Any]) -> bool:
    """Return whether the ligands belonging to the experiment need to be checked for contaminants.

    Args:
        row: The row containing the model and experiment information.
        contaminant: Specification of the contaminant to check for.

    Returns:
        True if the row needs to be checked for contaminants, False otherwise.
    """
    needs_check = _check_row_match(row, contaminant)

    if "exclude" in contaminant:
        needs_check = needs_check and not _check_row_match(row, contaminant["exclude"])

    return needs_check


def _annotate_single_experiments(
    row: pd.Series,
    contaminants: list[dict[str, Any]],
    allele2reference_model: dict[str, DeconvolutionModelMHC1],
) -> pd.DataFrame:
    """Annotate a single experiment with contaminant information.

    Args:
        row: The row containing the model and experiment information.
        contaminants: List of contaminant specifications.
        allele2reference_model: Dictionary mapping alleles to their corresponding reference
            deconvolution models.

    Returns:
        DataFrame containing the peptides annotated with contaminant information.
    """
    model_path = AnyPath(row["model_path"])
    model = DeconvolutionModelMHC1.load(model_path)

    df_peptides = load_csv(model_path / "responsibilities.csv")
    df_peptides["group"] = row["group"]
    df_peptides["experiment_id"] = row["experiment_id"]

    # only compute scores if needed for at least one contaminant type
    scores = None

    for contaminant in contaminants:
        column = f"contaminant_{contaminant['identifier']}"

        if not row[column]:
            df_peptides[column] = False
            continue

        if scores is None:
            scores = model.predict(df_peptides["peptide"])["score"]

        contaminant_allele = contaminant["contaminant_allele"]
        contaminant_model = allele2reference_model[contaminant_allele]
        contaminant_scores = contaminant_model.predict(df_peptides["peptide"])["score"]

        df_peptides[column] = scores < contaminant_scores

    return df_peptides


def _log_contaminant_counts(
    df_peptides: pd.DataFrame,
    contaminants: list[dict[str, Any]],
) -> None:
    """Log the counts of contaminant peptides.

    Args:
        df_peptides: DataFrame containing the peptide data.
        contaminants: List of contaminant specifications.
    """
    columns = [f"contaminant_{contaminant['identifier']}" for contaminant in contaminants] + [
        "contaminant_any"
    ]
    identifiers = [contaminant["identifier"] for contaminant in contaminants] + ["any"]

    for column, identifier in zip(columns, identifiers):
        value_counts = df_peptides[column].value_counts().to_dict()

        number_contaminated = value_counts.get(True, 0)
        number_non_contaminated = value_counts.get(False, 0)
        log.info(
            f"Contaminant {identifier}:"
            f" {number_contaminated:,} / {number_non_contaminated:,} contaminated"
            f" ({number_contaminated / (number_contaminated + number_non_contaminated) * 100:.2f}"
            " %)"
        )
