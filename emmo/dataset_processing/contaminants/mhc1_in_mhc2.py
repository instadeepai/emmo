"""Identification and annotation of potential MHC1 contaminant peptides in MHC2 ligandomics data."""
from __future__ import annotations

from typing import Any

import pandas as pd
from cloudpathlib import AnyPath

from emmo.io.file import load_csv
from emmo.io.file import Openable
from emmo.models.deconvolution import DeconvolutionModelMHC2
from emmo.pipeline.model_selection import get_best_run_details
from emmo.utils import logger

log = logger.get(__name__)

MHC1_CONTAMINANT_LENGTH_RANGE = (8, 12)


def annotate_mhc1_contaminants_in_mhc2_ligandomics(
    model_path: Openable | None = None,
    df_models: pd.DataFrame | None = None,
    additional_columns_to_keep: list[str] | None = None,
) -> pd.DataFrame:
    """Annotate potential MHC1 contaminants in the peptides of MHC2 deconvolution models.

    The input can be either a model path or a DataFrame with model paths and other information. The
    function annotates the peptides in the `responsibilities.csv` files of MHC2 deconvolution models
    with a boolean indicating whether the peptide is a potential MHC1 contaminant. It is True if the
    following conditions are satisfied:
    - The length of the peptide is in the MHC1 length range [8, 12]
    - The peptide is not assigned to the flat motif
    - The peptide is assigned to a motif with
        - peptide length mode is <= 9, or
        - peptide length mode of 10 and the offset is shifted by +1 towards the C-terminus (this
          covers the cases of MHC1 9-mer motifs that cover the main anchor positions in 10 mers,
          i.e., positions 2, 3 and 10)

    The function returns a DataFrame with the following columns:
    - peptide: The peptide sequence.
    - number_of_classes: The number of classes in the respective model.
    - best_class: The best class of the peptide.
    - mhc1_contaminant_annotation: MHC1 contaminant annotation.

    In case 'df_models' is provided, the columns 'number_of_classes', 'best_class' and
    'mhc1_contaminant_annotation' will be lists aggregated across the models of the same group. The
    DataFrame will additionally contain the following columns:
    - group: The group of the model.
    - mhc1_contaminant_best_aic: The MHC1 contaminant annotation for the model with the best AIC.
    - mhc1_contaminant_annotation_consensus: The consensus MHC1 contaminant annotation. This is
        True if the model with the best AIC and all models with higher number of classes have a
        positive contaminant annotation.
    - any additional columns specified in the additional_columns_to_keep parameter.

    Args:
        model_path: Path to the model directory.
        df_models: DataFrame with model paths and other information. The DataFrame should contain
            the column 'group', 'model_path', 'number_of_classes, and columns specified in the
            additional_columns_to_keep parameter.
        additional_columns_to_keep: List of additional columns from df_models to keep in the
            output DataFrame.

    Returns:
        DataFrame with the peptides, their best class, a boolean indicating whether the class
        is a potential MHC1 contaminant, and any specified additional columns.

    Raises:
        ValueError: If neither model_path nor df_models is provided, or if both are provided.
        ValueError: If any of the additional columns to keep are not found in df_models.
        TypeError: If a DataFrame is provided as model_path.
    """
    if model_path is None and df_models is None:
        raise ValueError("Either model_path or df_models must be provided.")
    if model_path is not None and df_models is not None:
        raise ValueError("Only one of model_path or df_models can be provided.")
    if isinstance(model_path, pd.DataFrame):
        raise TypeError("Dataframe provided as model_path. Use df_models instead.")

    if model_path is not None:
        return _annotate_mhc1_contaminants_in_mhc2_ligandomics(model_path)
    else:
        return _annotate_mhc1_contaminants_in_mhc2_ligandomics_consensus(
            df_models, additional_columns_to_keep
        )


def _annotate_mhc1_contaminants_in_mhc2_ligandomics(model_path: Openable) -> pd.DataFrame:
    """Annotate potential MHC1 contaminants in the peptides of an MHC2 deconvolution model.

    See the function `annotate_mhc1_contaminants_in_mhc2_ligandomics` for details.

    Args:
        model_path: Path to the model directory.

    Returns:
        DataFrame with the peptides, their best class, the number of classes, and a boolean
        indicating whether the class is a potential MHC1 contaminant.
    """
    df_responsibilities = load_csv(AnyPath(model_path) / "responsibilities.csv")

    df_responsibilities["best_class"] = df_responsibilities["best_class"].astype(str)
    df_responsibilities["peptide_length"] = df_responsibilities["peptide"].str.len()

    # get the mode of peptide length for each class
    modes = (
        df_responsibilities.groupby("best_class")["peptide_length"]
        .agg(lambda x: x.mode().values[0])
        .to_dict()
    )

    model = DeconvolutionModelMHC2.load(model_path)
    class2best_offset = model.best_offset_per_class()
    class2contaminant = {}
    for class_, mode in modes.items():
        # flat motif is not considered for potential MHC1 contaminant at the moment
        if class_ == "flat":
            class2contaminant[class_] = False
        # mode is <= 9
        elif mode <= 9:
            class2contaminant[class_] = True
            log.info(f"Detected potential MHC1 motif for 9mers: motif {class_} in {model_path}")
        # mode is 10 and offset is shifted by 1 towards the C-terminus
        elif mode == 10 and class2best_offset[class_] == 1:
            class2contaminant[class_] = True
            log.info(f"Detected potential MHC1 motif for 10mers: motif {class_} in {model_path}")
        else:
            class2contaminant[class_] = False

    df_responsibilities["mhc1_contaminant_annotation"] = df_responsibilities["best_class"].map(
        class2contaminant
    )
    df_responsibilities["number_of_classes"] = model.number_of_classes

    # only peptides in the MHC1 length range are considered
    df_responsibilities["mhc1_contaminant_annotation"] = df_responsibilities[
        "mhc1_contaminant_annotation"
    ] & df_responsibilities["peptide_length"].between(
        *MHC1_CONTAMINANT_LENGTH_RANGE, inclusive="both"
    )

    return df_responsibilities[
        ["peptide", "number_of_classes", "best_class", "mhc1_contaminant_annotation"]
    ]


def _annotate_mhc1_contaminants_consensus_single_row(row: pd.Series) -> tuple[bool, bool]:
    """Get the MHC1 contaminant annotation.

    See the function `annotate_mhc1_contaminants_in_mhc2_ligandomics` for details.

    Args:
        row: Row of the DataFrame with the best AIC model and the contaminant annotations of all
            models.

    Returns:
        True if the consensus contaminant annotation is positive, False otherwise.
    """
    # assumes that the elements in the list are ordered by number of classes
    number_of_classes = row["number_of_classes"]
    best_model = row["number_of_classes_best_aic"]
    annotations = row["mhc1_contaminant_annotation"]

    start_idx = number_of_classes.index(best_model)

    return annotations[start_idx], all(annotations[start_idx:])


def _annotate_mhc1_contaminants_in_mhc2_ligandomics_consensus(
    df_models: pd.DataFrame,
    additional_columns_to_keep: list[str] | None = None,
) -> pd.DataFrame:
    """Get the consensus annotation for potential MHC1 contaminants over models.

    The functions considers the model with the best AIC and checks whether this model and all
    available models with higher number of classes have a positive contaminant annotation. Only then
    will the consensus contaminant annotation be positive.

    The function returns a DataFrame with the following columns:
    - group: The group of the model.
    - peptide: The peptide sequence.
    - number_of_classes: List of numbers of classes (aggregated across the models).
    - best_class: List of corresponding best classes of the peptide (aggregated across the models).
    - mhc1_contaminant_annotation: List of corresponding MHC1 contaminant annotations (aggregated
        across the models).
    - any additional columns specified in the additional_columns_to_keep parameter.

    Args:
        df_models: DataFrame with model paths and other information. The DataFrame should contain
            the columns 'group', 'model_path', 'number_of_classes, and columns specified in the
            additional_columns_to_keep parameter.
        additional_columns_to_keep: List of additional columns from df_models to keep in the
            output DataFrame.

    Returns:
        DataFrame with peptides and their MHC1 contaminant annotation.
    """
    if additional_columns_to_keep is None:
        additional_columns_to_keep = []
    required_columns = ["model_path", "group", "number_of_classes"] + additional_columns_to_keep
    missing_columns = set(required_columns) - set(df_models.columns)
    if missing_columns:
        raise ValueError(f"Columns {sorted(missing_columns)} not found in df_models.")

    if df_models.duplicated(subset=["group", "number_of_classes"]).any():
        raise ValueError(
            "DataFrame contains duplicates for the same 'group' and 'number_of_classes'"
        )

    df_models = df_models.copy()

    log.info(f"Loading details of best run for {len(df_models):,} models ...")

    df_models["best_run"] = df_models["model_path"].apply(get_best_run_details)
    df_models["aic_pssm"] = df_models["best_run"].apply(lambda x: x["AIC_PSSM"])

    idx_best_models = df_models.groupby("group")["aic_pssm"].idxmin()
    group2best_number_of_classes = (
        df_models.loc[idx_best_models].set_index("group")["number_of_classes"].to_dict()
    )

    log.info(f"Annotating {len(df_models):,} models for potential MHC1 contaminants ...")

    dfs = []
    for _, row in df_models.iterrows():
        df_single = _annotate_mhc1_contaminants_in_mhc2_ligandomics(AnyPath(row["model_path"]))
        for col in ["group"] + additional_columns_to_keep:
            df_single[col] = row[col]
        dfs.append(df_single)

    log.info("Annotation finished. Aggregating results ...")

    agg_funcs: dict[str, Any] = {
        "number_of_classes": list,
        "best_class": list,
        "mhc1_contaminant_annotation": list,
    }
    agg_funcs.update({col: "first" for col in additional_columns_to_keep})

    df_annotated = (
        pd.concat(dfs)
        .sort_values(["group", "peptide", "number_of_classes"])
        .groupby(["group", "peptide"])
        .agg(agg_funcs)
        .reset_index()
    )

    log.info(
        f"Aggregated results for {len(df_annotated):,} peptides across "
        f"{df_models['group'].nunique():,} models. Computing consensus ..."
    )

    df_annotated["number_of_classes_best_aic"] = df_annotated["group"].map(
        group2best_number_of_classes
    )

    df_annotated[
        ["mhc1_contaminant_best_aic", "mhc1_contaminant_annotation_consensus"]
    ] = df_annotated.apply(
        _annotate_mhc1_contaminants_consensus_single_row,
        axis=1,
        result_type="expand",
    )

    counts = df_annotated.groupby("mhc1_contaminant_annotation_consensus").size().to_dict()
    log.info(
        f"Consensus MHC1 contaminant annotation: {counts.get(True, 0):,} true, "
        f"{counts.get(False, 0):,} false"
    )

    return df_annotated
