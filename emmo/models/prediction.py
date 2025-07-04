"""Binding prediction using the output of EM deconvolution and motif extraction.

This predictor is similar to MixMHC2pred (Racle et al. 2019).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from cloudpathlib import AnyPath

from emmo.constants import MHC2_ALPHA_COL
from emmo.constants import MHC2_BETA_COL
from emmo.io.file import load_csv
from emmo.io.file import load_json
from emmo.io.file import Openable
from emmo.io.file import save_json
from emmo.io.output import write_matrix
from emmo.models.cleavage import CleavageModel
from emmo.models.deconvolution import DeconvolutionModelMHC2
from emmo.pipeline.background import Background
from emmo.pipeline.background import BackgroundType
from emmo.pipeline.model_selection import SelectedModel
from emmo.pipeline.sequences import SequenceManager
from emmo.resources.length_distribution import get_length_distribution
from emmo.utils import logger
from emmo.utils.alleles import parse_mhc2_allele_pair
from emmo.utils.exceptions import NoSequencesError
from emmo.utils.motifs import information_content
from emmo.utils.motifs import position_probability_matrix
from emmo.utils.offsets import AlignedOffsets
from emmo.utils.sequence_distance import nearest_neighbors

log = logger.get(__name__)


class PredictorMHC2:
    """MHC2 binding predictor."""

    def __init__(
        self,
        alphabet: str | tuple[str, ...] | list[str],
        ppms: dict[str, np.ndarray],
        cleavage_model_path: Openable,
        offset_weights: np.ndarray,
        background: BackgroundType,
        motif_length: int = 9,
        length_distribution: dict[int, float] | str | None = None,
        compilation_details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the predictor.

        Args:
            alphabet: The (amino acid) alphabet.
            ppms: The position probability matrices.
            cleavage_model_path: Path to the cleavage model.
            offset_weights: Offset weights.
            background: The background amino acid frequencies. Must be a string corresponding to
                one of the available backgrounds.
            motif_length: Motif length.
            length_distribution: The length distribution of the ligands. Can be a string
                corresponding to one of the available distributions.
            compilation_details: Details about the model compilation.
        """
        self.alphabet = alphabet
        self.alphabet_index = {a: i for i, a in enumerate(alphabet)}

        self.background = Background(background)

        self.ppms = ppms
        self.available_alleles = sorted(self.ppms)
        self.cleavage_model = CleavageModel.load(cleavage_model_path)
        self.offset_weights = offset_weights
        self.motif_length = motif_length

        if isinstance(length_distribution, str):
            self.length_distribution = get_length_distribution(length_distribution)
        else:
            self.length_distribution = length_distribution

        max_seq_length = len(self.offset_weights) + self.motif_length - 1
        self.aligned_offsets = AlignedOffsets(self.motif_length, max_seq_length)
        if self.aligned_offsets.n_offsets != len(self.offset_weights):
            raise ValueError("number of offsets weights does not match")

        self.compilation_details = {}
        if compilation_details is not None:
            self.compilation_details.update(compilation_details)

    @classmethod
    def load(cls, directory: Openable) -> PredictorMHC2:
        """Load a model from a directory.

        Args:
            directory: The directory containing the model files.

        Returns:
            The loaded model.
        """
        directory = AnyPath(directory)

        model_specs = load_json(directory / "model_specs.json")

        ppms = {}
        for file in (directory / "binding").iterdir():
            if file.is_file() and file.suffix == ".csv":
                try:
                    allele_alpha, allele_beta = parse_mhc2_allele_pair(file.stem)
                    allele = f"{allele_alpha}-{allele_beta}"
                except ValueError:
                    log.warning(
                        f"could not parse alpha and beta allele from file name '{file.name}'"
                    )
                    allele = file.stem

                matrix = load_csv(file, index_col=0, header=0).to_numpy()
                ppms[allele] = matrix

        if "length_distribution" in model_specs and isinstance(
            model_specs["length_distribution"], dict
        ):
            length_distribution = {
                int(length): x for length, x in model_specs["length_distribution"].items()
            }
        else:
            length_distribution = model_specs.get("length_distribution", None)

        if "background" not in model_specs:
            default_background = "MHC2_biondeep"
            log.info(
                "Loading model without saved background, "
                f"will use default '{default_background}' for older models"
            )
            model_specs["background"] = default_background

        return cls(
            model_specs["alphabet"],
            ppms,
            directory / "cleavage",
            np.asarray(model_specs["offset_weights"]),
            model_specs["background"],
            motif_length=model_specs["motif_length"],
            length_distribution=length_distribution,
            compilation_details=model_specs.get("compilation_details"),
        )

    @classmethod
    def compile_from_selected_models(
        cls,
        selected_models: dict[str, SelectedModel],
        cleavage_model_path: Openable,
        background: BackgroundType | None,
        peptides_path: Openable | None = None,
        motif_length: int = 9,
        length_distribution: str | None = None,
        recompute_ppms_from_best_responsibility: bool = False,
    ) -> PredictorMHC2:
        """Compile a predictor from deconvolution models and a cleavage model.

        Args:
            selected_models: The deconvolution models and classes within these models to be
                gathered. The keys are the alleles. The values are dicts containing at least the
                following keys:
                - 'classes': Number of classes in the deconvolution model.
                - 'motif': The motif to select from the deconvolution model (1-based index).
                - 'model_path': The path to the deconvolution model directory.
            cleavage_model_path: The path to the cleavage model.
            background: The background amino acid frequencies. Can also be a string corresponding
                to one of the available backgrounds. If this is None, the distribution will be
                taken from one of the deconvolution models.
            peptides_path: The directory containing the peptide files. This is only needed for
                older deconvolution models that did not save the 'training_params'.
            motif_length: The motif length.
            length_distribution: The length distribution of the ligands. Must be a string
                corresponding to  one of the available distributions (or None).
            recompute_ppms_from_best_responsibility: Whether to recompute the PPMs from the core
                predictions in the deconvolution runs. The 'responsibilities.csv' file must be
                contained in the respective deconvolution models directories.

        Returns:
            The compiled MHC2 binding predictor.
        """
        if not selected_models:
            raise ValueError("'selected_models' must not be empty")

        loaded_binding_models = {}
        selection_details = {}
        ppms = {}

        for allele, selection in selected_models.items():
            model = DeconvolutionModelMHC2.load(selection["model_path"])
            loaded_binding_models[allele] = model
            selection_details[allele] = selection.copy()
            del selection_details[allele]["model_path"]

            motif_idx = selection["motif"]

            if recompute_ppms_from_best_responsibility:
                try:
                    ppms[allele] = cls._recompute_ppms(selection["model_path"], motif_idx)
                except NoSequencesError:
                    log.warning(
                        f"Could not recompute PPM for allele {allele} because no peptides were "
                        "assigned to the class, falling back to PPM from deconvolution model"
                    )

            if allele not in ppms:
                ppms[allele] = model.ppm[motif_idx - 1].copy()

        averaged_offset_weights = cls._compute_averaged_offset_weights(
            loaded_binding_models, selection_details, peptides_path=peptides_path
        )

        if background is None:
            # pick an arbitrary but deterministic allele from the available ones
            allele = sorted(loaded_binding_models)[0]
            background = loaded_binding_models[allele].background
            log.info(
                "Using background distribution associated with deconvolution model for allele "
                f"{allele}"
            )

        return cls(
            next(iter(loaded_binding_models.values())).alphabet,
            ppms,
            cleavage_model_path,
            averaged_offset_weights,
            background,
            motif_length=motif_length,
            length_distribution=length_distribution,
            compilation_details={"selected_motifs": selection_details},
        )

    @classmethod
    def _recompute_ppms(cls, model_path: Openable, motif: int) -> np.ndarray:
        """Recompute PPMs from the core predictions.

        Args:
            model_path: The path to the deconvolution model directory.
            motif: The motif to select from the deconvolution model (1-based index).

        Raises:
            NoSequencesError: If no sequences were assigned to the motif based on the maximum
                responsibility.

        Returns:
            The position probability matrix.
        """
        model_path = AnyPath(model_path)
        resp = load_csv(model_path / "responsibilities.csv")

        sequences = list(
            resp.loc[resp["best_class"].astype(str) == str(motif), "binding_core_prediction"]
        )

        if not sequences:
            raise NoSequencesError(
                "Could not compute PPM because no peptides were assigned to the class"
            )

        return position_probability_matrix(sequences, use_pseudocounts=True, pseudocount_beta=200)

    @classmethod
    def _compute_averaged_offset_weights(
        cls,
        loaded_binding_models: dict[str, DeconvolutionModelMHC2],
        selection_details: dict[str, SelectedModel],
        peptides_path: Openable | None = None,
    ) -> np.ndarray:
        """Weighted average of the offset weights over all models.

        The function also adds 'effective_peptide_number' and 'motif weight' for each allele in
        'selection_details'.

        Args:
            loaded_binding_models: The loaded binding models.
            selection_details: A dict containing the alleles as keys and as values dicts with at
                at least the following keys:
                - 'classes': Number of classes in the deconvolution model.
                - 'motif': The motif to select from the deconvolution model (1-based index).
            peptides_path: The directory containing the peptide files. This is only needed for
                older deconvolution models that did not save the 'training_params'.

        Raises:
            RuntimeError: If the effective peptide counts could not be obtained from the loaded
                models.

        Returns:
            The averaged offset weights.
        """
        effective_peptide_counts_per_allele = {}

        # if peptides path is provided, recompute the effective peptide count from there
        if peptides_path is not None:
            for file in AnyPath(peptides_path).iterdir():
                allele = file.stem
                sm = SequenceManager.load_from_txt(file)
                effective_peptide_counts_per_allele[allele] = np.sum(sm.get_similarity_weights())
        # newer models have the effective training peptide count in their associated training
        # parameters
        else:
            try:
                effective_peptide_counts_per_allele = {
                    allele: model.training_params["effective_peptide_number"]
                    for allele, model in loaded_binding_models.items()
                }
            except KeyError:
                raise RuntimeError(
                    "could not find effective allele weight in model params, "
                    "try providing the peptides_path"
                )

        averaged_weights = np.asarray([0.0], dtype=np.float64)

        for allele, model in loaded_binding_models.items():
            eff_num_peptides = effective_peptide_counts_per_allele[allele]
            weights = model.class_weights[selection_details[allele]["motif"] - 1]

            selection_details[allele]["effective_peptide_number"] = eff_num_peptides
            selection_details[allele]["motif_weight"] = weights.sum()

            weights *= eff_num_peptides

            # give alias such a is not shorter than b
            if len(averaged_weights) >= len(weights):
                a, b = averaged_weights, weights
            else:
                a, b = weights, averaged_weights

            # add the two array in an aligned manner
            offset_in_a = (len(a) - len(b)) // 2
            a[offset_in_a : offset_in_a + len(b)] += b
            averaged_weights = a

        return averaged_weights / np.sum(averaged_weights)

    def save(self, directory: Openable, force: bool = False) -> None:
        """Save the model to a directory.

        Args:
            directory: The directory where to save the model.
            force: Whether to also write the model files if the directory already exists.
        """
        directory = AnyPath(directory)
        directory_binding = directory / "binding"
        directory_cleavage = directory / "cleavage"

        model_specs = {
            x: self.__dict__[x] for x in ["alphabet", "motif_length", "length_distribution"]
        }
        model_specs["background"] = self.background.get_representation()
        model_specs["offset_weights"] = self.offset_weights.tolist()
        if self.compilation_details:
            model_specs["compilation_details"] = self.compilation_details

        save_json(model_specs, directory / "model_specs.json", force=force)

        for allele, ppm in self.ppms.items():
            write_matrix(directory_binding / f"{allele}.csv", ppm, self.alphabet, force=force)

        self.cleavage_model.save(directory_cleavage, force=force)

    def _score_binding(
        self,
        peptide: str,
        ppm: np.ndarray,
        use_offset_weight: bool = True,
        use_background: bool = True,
        info_content: np.ndarray | None = None,
    ) -> tuple[float, int]:
        """Score a peptide.

        The calculates the sum of scores over all classes and all possible offsets. For each
        offset, the score is the product of the respective values from the PPM or PSSM, which is
        additionally multiplied by the offset weight. Alternatively, if 'info_content' is provided,
        then the dot product of the values in the PPM/PSSM and the information content of the
        respective positions.

        Args:
            peptide: The peptide to be scored.
            ppm: Position-probability matrix.
            use_offset_weight: Whether to use the offset weights.
            use_background: Whether to use PSSMs, i.e., the PPM divided by background.
            info_content: The information content vector. If this is provided, an alternative
                scoring method per offset based on a dot product is used.

        Returns:
            The score and the best offset.
        """
        score = 0.0
        best_prob = float("-inf")
        best_offset = -1
        bg_freqs = self.background.frequencies

        for i, o in enumerate(self.aligned_offsets.get_offset_list(len(peptide))):
            # if peptide is longer than max_sequence_length, the following could happen
            if o < 0 or o >= len(self.offset_weights):
                continue

            aa_indices = [
                (k, self.alphabet_index[peptide[i + k]])
                for k in range(self.motif_length)
                # support also unknown characters like 'X'
                if peptide[i + k] in self.alphabet_index
            ]

            if use_background:
                prob = np.array([ppm[k, a] / bg_freqs[a] for k, a in aa_indices])
            else:
                prob = np.array([ppm[k, a] for k, a in aa_indices])

            if info_content is not None:
                prob *= np.array([info_content[k] for k, _ in aa_indices])
                prob = np.sum(prob)
            else:
                prob = np.prod(prob)

            if use_offset_weight:
                prob *= self.offset_weights[o]

            score += prob

            if prob > best_prob:
                best_prob = prob
                best_offset = i

        return score, best_offset

    # TODO: the following function needs refactoring
    def score_dataframe(  # noqa: CCR001
        self,
        df: pd.DataFrame,
        peptide_column: str = "peptide",
        allele_alpha_column: str = MHC2_ALPHA_COL,
        allele_beta_column: str = MHC2_BETA_COL,
        column_prefix: str = "emmo",
        inplace: bool = False,
        score_length: bool = False,
        pan_allelic: bool | str = False,
        use_offset_weight: bool = True,
        use_background: bool = True,
        use_information_content: bool = False,
    ) -> pd.DataFrame:
        """Score the peptide-allele pairs in a dataframe.

        Args:
            df: The dataframe.
            peptide_column: Name of the column containing the peptides.
            allele_alpha_column: Name of the column containing the allele alpha chain.
            allele_beta_column: Name of the column containing the allele beta chain.
            column_prefix: The prefix of the result columns.
            inplace: Whether to add the result columns in the original dataframe or first copy the
                dataframe.
            score_length: Whether to all include the scoring columns that use the length
                distribution.
            pan_allelic: Whether and which pan-allelic mode shall be used.
            use_offset_weight: Whether to use the offset weights.
            use_background: Whether to use PSSMs, i.e., the PPM divided by background.
            use_information_content: Whether to use an alternative scoring method per offset based
                on a dot product.

        Raises:
            NotImplementedError: If the specified pan-allelic mode is not yet available but planned.
            ValueError: If the specified pan-allelic mode is not available.

        Returns:
            The dataframe with the added result columns.
        """
        if not inplace:
            df = df.copy()

        prefix = column_prefix + "_" if column_prefix else ""

        # prefer originally available motifs whenever available
        used_ppms = {allele: [ppm] for allele, ppm in self.ppms.items()}

        alleles = df[allele_alpha_column] + "-" + df[allele_beta_column]
        allele_set = set(alleles.dropna().unique())
        unseen_alleles = sorted(allele_set - set(self.available_alleles))

        print(
            f"--------------------------------------------------------------\n"
            f"Running peptide-MHC2 binding prediction\n"
            f"--------------------------------------------------------------\n"
            f"Number of rows: {len(df)}\n"
            f"Number of different alleles: {len(allele_set)}\n"
            f"    Thereof unseen in training {len(unseen_alleles)} "
            f"({alleles.isin(unseen_alleles).sum()} rows)\n"
            f"Use offset weights: {use_offset_weight}\n"
            f"Use background: {use_background}\n"
            f"Use length distribution: {score_length}\n"
            f"Use information content scoring mode: {use_information_content}\n"
            f"Prefix for result columns: {column_prefix}\n"
            f"Pan-allelic mode: {pan_allelic}"
        )

        # add PPMs for additional alleles to make the predictor pan-allelic
        if pan_allelic == "nearest":
            print("    Using the following mapping " "[unseen allele --> seen allele(s)]:")

            neighbors = nearest_neighbors(self.available_alleles, unseen_alleles, "II")

            added_alleles_counter = 0
            for unseen_allele, (mapped_alleles, distance) in neighbors.items():
                if mapped_alleles:
                    print(
                        f'    {unseen_allele} --> {" ".join(mapped_alleles)} '
                        f"(distance {distance:.4f})"
                    )
                    added_alleles_counter += 1
                    used_ppms[unseen_allele] = [self.ppms[a] for a in mapped_alleles]
                else:
                    print(f"    {unseen_allele} could not be mapped")
            print(
                f"    Could add {added_alleles_counter} out of "
                f"{len(unseen_alleles)} for scoring"
            )

        elif pan_allelic == "predicted":
            raise NotImplementedError("predicted motifs/PPMs might be available in the future")

        elif pan_allelic is not False:
            raise ValueError(f"pan-allelic mode {pan_allelic} is not available")

        print("--------------------------------------------------------------")

        # pre-compute information content
        info_content_lookup = {
            allele: [
                (
                    information_content(ppm, self.background.frequencies)
                    if use_information_content
                    else None
                )
                for ppm in ppms
            ]
            for allele, ppms in used_ppms.items()
        }

        # now start the scoring
        binding_scores = []
        binding_offsets = []
        binding_cores = []

        for peptide, allele in zip(df[peptide_column], alleles):
            if allele not in used_ppms:
                binding_scores.append(np.nan)
                binding_offsets.append(np.nan)
                binding_cores.append(np.nan)
                continue

            # there may be more than one PPM (in "nearest neighbor" mode)
            _scores_and_offsets = [
                self._score_binding(
                    peptide,
                    ppm,
                    use_offset_weight=use_offset_weight,
                    use_background=use_background,
                    info_content=info_content,
                )
                for ppm, info_content in zip(used_ppms[allele], info_content_lookup[allele])
            ]
            _scores = [s for s, _ in _scores_and_offsets]

            # use mean score
            binding_scores.append(np.mean(_scores))

            # use core offset for which score is max
            offset = _scores_and_offsets[np.argmax(_scores)][1]
            binding_offsets.append(offset)
            binding_cores.append(peptide[offset : offset + self.motif_length])

        df[f"{prefix}binding"] = binding_scores
        df[f"{prefix}offset"] = binding_offsets
        df[f"{prefix}core"] = binding_cores

        df[f"{prefix}cleavage"] = self.cleavage_model.predict(df[peptide_column])

        df[f"{prefix}BC"] = df[f"{prefix}binding"] * df[f"{prefix}cleavage"]

        if score_length:
            if self.length_distribution is None:
                raise RuntimeError("length distribution is not available in this predictor")

            df[f"{prefix}length_score"] = (
                df[peptide_column].str.len().apply(lambda x: self.length_distribution.get(x, 0.0))
            )
            df[f"{prefix}BCL"] = df[f"{prefix}BC"] * df[f"{prefix}length_score"]
            df[f"{prefix}BL"] = df[f"{prefix}binding"] * df[f"{prefix}length_score"]

        return df
