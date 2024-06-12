"""Tensorflow implementation of the EM algorithm for MHC2 ligands."""
from __future__ import annotations

import tensorflow as tf

from emmo.em.mhc2_base import BaseEMRunnerMHC2
from emmo.pipeline.background import BackgroundType
from emmo.pipeline.sequences import SequenceManager


@tf.function
def _log_likelihood(
    pwm: tf.Tensor,
    sequence_indices: tf.Tensor,
    class_weights: tf.Tensor,
    offset_weight_indices: tf.Tensor,
    similarity_weight: tf.Tensor,
    offset_mask: tf.Tensor,
    ppm: tf.Tensor,
    pseudocount: tf.Tensor,
) -> tf.Tensor:
    """Compute the log-likelihood under the current model.

    The input PWM can contain either amino acid frequencies or log odds ratios w.r.t. some
    background. The latter is used in the EM algorithm (like in MoDec).

    Args:
        pwm: Position probability or scoring matrix.
        sequence_indices: Index tensor necessary to collect the probabilities from the PWM
            according to the sequences.
        class_weights: Class and offset weights.
        offset_weight_indices: Maps between aligned offsets and right-padded offsets.
        similarity_weight: Sequence weights.
        offset_mask: Mask for right-padded offset positions.
        ppm: Position probability matrix.
        pseudocount: Pseudocount for prior computation.

    Returns:
        Log likelihood.
    """
    # Collect the probabilities from PWM according to the sequences. The resulting tensor will be
    # right-padded in the offsets dimension.
    # (sequences, classes, right-padded offsets, motif_length)
    x = tf.gather_nd(params=pwm, indices=sequence_indices)

    # Product along the motif length dimension.
    # (sequences, classes, right-padded offsets)
    x = tf.math.reduce_prod(x, axis=3)

    # Collect the corresponding class and offset weights such that they are also right-padded.
    # (sequences, classes, right-padded offsets)
    x = x * tf.gather_nd(params=class_weights, indices=offset_weight_indices)

    # Mask the padding positions.
    # (sequences, classes, right-padded offsets)
    x = x * offset_mask

    # Sum over all classes and offsets.
    # (sequences,)
    x = tf.reduce_sum(x, axis=[1, 2])

    # Apply the sequence weights.
    # (,)
    x = tf.reduce_sum(similarity_weight * tf.math.log(x))

    # Finally add the prior log probability (this is similar to what MixMHCp does, but should be
    # looked at but someone with expertise in mixture models).
    # (,)
    x = x + pseudocount * tf.reduce_sum(tf.math.log(ppm[:-1]))

    return x


@tf.function
def _expectation(
    pwm: tf.Tensor,
    sequence_indices: tf.Tensor,
    class_weights: tf.Tensor,
    offset_weight_indices: tf.Tensor,
    responsibilities: tf.Tensor,
    responsibility_indices: tf.Tensor,
    offset_mask: tf.Tensor,
) -> tf.Tensor:
    """Expectation step.

    Args:
        pwm: Position probability or scoring matrix.
        sequence_indices: Index tensor necessary to collect the probabilities from the PWM
            according to the sequences.
        class_weights: Class and offset weights.
        offset_weight_indices: Maps between aligned offsets and right-padded offsets.
        responsibilities: The current responsibilities.
        responsibility_indices: Maps between aligned offsets and right-padded offsets.
        offset_mask: Mask for right-padded offset positions.

    Returns:
        The new responsibilities.
    """
    # Collect the probabilities from PWM according to the sequences. The resulting tensor will be
    # right-padded in the offsets dimension.
    # (sequences, classes, right-padded offsets, motif_length)
    x = tf.gather_nd(params=pwm, indices=sequence_indices)

    # Product along the motif length dimension.
    # (sequences, classes, right-padded offsets)
    x = tf.math.reduce_prod(x, axis=3)

    # Collect the corresponding class and offset weights such that they are also right-padded.
    # (sequences, classes, right-padded offsets)
    x = x * tf.gather_nd(params=class_weights, indices=offset_weight_indices)

    # Mask the padding positions.
    # (sequences, classes, right-padded offsets)
    x = x * offset_mask

    # At this point, we have the likelihood for each sequence, class and offset in a such manner
    # that all possible offsets for an individual sequence are right-padded. So it remains to move
    # them to the correct aligned offsets within the offsets dimension.

    # (sequences, classes, aligned offsets)
    x = tf.scatter_nd(indices=responsibility_indices, updates=x, shape=responsibilities.shape)

    # Normalize such that responsibilities per sequence sum up to one.
    # (sequences, classes, aligned offsets)
    return x / tf.reduce_sum(x, axis=(1, 2), keepdims=True)


@tf.function
def _maximization(
    responsibilities: tf.Tensor,
    responsibility_indices: tf.Tensor,
    sequence_indices: tf.Tensor,
    similarity_weight: tf.Tensor,
    offset_mask: tf.Tensor,
    pseudocount: tf.Tensor,
    background_freqs_broadcast: tf.Tensor,
    offset_upweighting: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Maximization step.

    Args:
        responsibilities: Responsibilities.
        responsibility_indices: Maps between aligned offsets and right-padded offsets.
        sequence_indices: Index tensor necessary to collect the probabilities from the PWM
            according to the sequences.
        similarity_weight: Sequence weights.
        offset_mask: Mask for right-padded offset positions.
        pseudocount: Pseudocount.
        background_freqs_broadcast: Broadcast-version of the background frequencies.
        offset_upweighting: Factors for correcting the bias for aligned offset position 0 (middle
            offset).

    Returns:
        The updated class and offset weights, PPM, and PSSM.
    """
    n_classes = responsibilities.shape[1]
    _, motif_length, n_alphabet = background_freqs_broadcast.shape

    # Weight responsibilities according to the sequence weights.
    # (sequences, classes, aligned offsets)
    x = responsibilities * similarity_weight[:, tf.newaxis, tf.newaxis]

    # Sum up, upweight the middle offset, and add pseudocounts.
    # (classes, aligned offsets)
    class_weights = tf.reduce_sum(x, axis=0) * offset_upweighting + pseudocount

    # Normalize so that class weights sum to one.
    # (classes, aligned offsets)
    class_weights = class_weights / tf.reduce_sum(class_weights, keepdims=True)

    # Collect the responsibilities to be right padded.
    # (sequences, classes, right-padded offsets)
    x = tf.gather_nd(params=x, indices=responsibility_indices)

    # Mask the padding positions.
    # (sequences, classes, right-padded offsets)
    x = x * offset_mask

    # Repeat tensor along a new motif length dimension such that it can be used with 'scatter_nd()'
    # in the next step.
    # (sequences, classes, right-padded offsets, motif_length)
    x = tf.broadcast_to(tf.expand_dims(x, axis=-1), shape=(*x.shape, motif_length))

    # Sum up (over all offsets) and add pseudocounts.
    # (classes, motif_length, alphabet length)
    x = (
        tf.scatter_nd(
            indices=sequence_indices, updates=x, shape=(n_classes, motif_length, n_alphabet)
        )
        + pseudocount
    )

    # Remove the flat motif and normalize so that frequencies sum to one for each position.
    # (classes, motif_length, alphabet length)
    x = x[:-1] / tf.reduce_sum(x[:-1], axis=2, keepdims=True)

    # Restore original flat motif.
    # (classes, motif_length, alphabet length)
    ppm = tf.concat(values=[x, background_freqs_broadcast], axis=0)

    # Recompute position-specific scoring matrix.
    # (classes, motif_length, alphabet length)
    pssm = ppm / background_freqs_broadcast

    return class_weights, ppm, pssm


@tf.function
def _maximization_and_likelihood(
    responsibilities: tf.Tensor,
    responsibility_indices: tf.Tensor,
    sequence_indices: tf.Tensor,
    offset_weight_indices: tf.Tensor,
    offset_mask: tf.Tensor,
    similarity_weights: tf.Tensor,
    pseudocount: tf.Tensor,
    background_freqs_broadcast: tf.Tensor,
    offset_upweighting: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Maximization step and log likelihood computation.

    Args:
        responsibilities: Responsibilities.
        responsibility_indices: Maps between aligned offsets and right-padded offsets.
        sequence_indices: Index tensor necessary to collect the probabilities from the PWM
            according to the sequences.
        offset_weight_indices: Maps between aligned offsets and right-padded offsets.
        offset_mask: Mask for right-padded offset positions.
        similarity_weight: Sequence weights.
        pseudocount: Pseudocount.
        background_freqs_broadcast: Broadcast-version of the background frequencies.
        offset_upweighting: Factors for correcting the bias for aligned offset position 0 (middle
            offset).

    Returns:
        The updated class and offset weights, PPM, PSSM, and the new log likelihood.
    """
    class_weights, ppm, pssm = _maximization(
        responsibilities,
        responsibility_indices,
        sequence_indices,
        similarity_weights,
        offset_mask,
        pseudocount,
        background_freqs_broadcast,
        offset_upweighting,
    )

    log_likelihood_pssm = _log_likelihood(
        pssm,
        sequence_indices,
        class_weights,
        offset_weight_indices,
        similarity_weights,
        offset_mask,
        ppm,
        pseudocount,
    )

    return class_weights, ppm, pssm, log_likelihood_pssm


@tf.function
def _run_expectation_maximization(
    min_error: tf.Tensor,
    responsibilities: tf.Tensor,
    responsibility_indices: tf.Tensor,
    sequence_indices: tf.Tensor,
    offset_weight_indices: tf.Tensor,
    offset_mask: tf.Tensor,
    similarity_weights: tf.Tensor,
    pseudocount: tf.Tensor,
    background_freqs_broadcast: tf.Tensor,
    offset_upweighting: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Run the expectation-maximization algorithm until convergence.

    Args:
        min_error: When the log likelihood difference between two steps becomes smaller than this
            value, the EM run is finished.
        responsibilities: Initial responsibilities.
        responsibility_indices: Maps between aligned offsets and right-padded offsets.
        sequence_indices: Index tensor necessary to collect the probabilities from the PWM
            according to the sequences.
        offset_weight_indices: Maps between aligned offsets and right-padded offsets.
        offset_mask: Mask for right-padded offset positions.
        similarity_weights: Sequence weights.
        pseudocount: Pseudocount.
        background_freqs_broadcast: Broadcast-version of the background frequencies.
        offset_upweighting: Factors for correcting the bias for aligned offset position 0 (middle
            offset).

    Returns:
        The final class and offset weights, PPM, PSSM, log likelihood based on PSSM and PPM,
        responsibilities, and the number of necessary EM steps.
    """
    # Initialize PPMs and class weights based on initial responsibilities.
    class_weights, ppm, pssm, log_likelihood_pssm = _maximization_and_likelihood(
        responsibilities,
        responsibility_indices,
        sequence_indices,
        offset_weight_indices,
        offset_mask,
        similarity_weights,
        pseudocount,
        background_freqs_broadcast,
        offset_upweighting,
    )

    log_likelihood_error = min_error + 1
    steps = 0

    while log_likelihood_error > min_error:
        steps += 1

        # Expectation step.
        responsibilities = _expectation(
            pssm,
            sequence_indices,
            class_weights,
            offset_weight_indices,
            responsibilities,
            responsibility_indices,
            offset_mask,
        )

        # Maximization step and update log likelihood.
        class_weights, ppm, pssm, new_log_likelihood = _maximization_and_likelihood(
            responsibilities,
            responsibility_indices,
            sequence_indices,
            offset_weight_indices,
            offset_mask,
            similarity_weights,
            pseudocount,
            background_freqs_broadcast,
            offset_upweighting,
        )

        log_likelihood_error = tf.math.abs(log_likelihood_pssm - new_log_likelihood)
        log_likelihood_pssm = new_log_likelihood

    log_likelihood_ppm = _log_likelihood(
        ppm,
        sequence_indices,
        class_weights,
        offset_weight_indices,
        similarity_weights,
        offset_mask,
        ppm,
        pseudocount,
    )

    return (
        class_weights,
        ppm,
        pssm,
        log_likelihood_pssm,
        log_likelihood_ppm,
        responsibilities,
        steps,
    )


class EMRunnerMHC2(BaseEMRunnerMHC2):
    """Class for running the EM algorithm for MHC2 ligands (TF version)."""

    def __init__(
        self,
        sequence_manager: SequenceManager,
        motif_length: int,
        number_of_classes: int,
        background: BackgroundType,
        tf_precision: str = "float64",
    ) -> None:
        """Initialize the MHC2 EM runner (tensorflow version).

        Args:
            sequence_manager: The instance holding the input sequences.
            motif_length: The length of the motif(s) to be estimated.
            number_of_classes: The number of motifs/classes to be identified (not counting the flat
                motif).
            background: The background amino acid frequencies. Can also be a string corresponding
                to one of the available backgrounds.
            tf_precision: Float precision to be used for the tensorflow-based operations.

        Raises:
            ValueError: If the specified float precision is not supported.
        """
        super().__init__(
            sequence_manager,
            motif_length,
            number_of_classes,
            background,
        )

        if tf_precision == "float64":
            self.tf_precision = tf.float64
        elif tf_precision == "float32":
            self.tf_precision = tf.float32
        else:
            raise ValueError(f"invalid tensorflow precision {tf_precision}")

        # Broadcast background frequencies version for concatenation.
        self.background_freqs_broadcast = tf.broadcast_to(
            tf.constant(self.background.frequencies, dtype=self.tf_precision),
            shape=(1, self.motif_length, self.background.frequencies.shape[0]),
        )

        # Convert similarity and offset weights to tf.Tensor
        self.similarity_weights = tf.constant(self.similarity_weights, dtype=self.tf_precision)
        self.offset_upweighting = tf.constant(self.offset_upweighting, dtype=self.tf_precision)

        self._compute_padding()
        self._compute_indices()

    def _compute_padding(self) -> None:
        """Initialize the padding for sequences and offsets, and the corresponding masks."""
        # Pad all sequences on the rights with zeros. This needs masking later because zero is also
        # a valid amino acid.
        self.padded_sequences = [
            seq + (self.sm.max_length + len(seq)) * [0] for seq in self.sequences
        ]

        # Pad all offset lists on the rights with zeros. This needs masking later because zero is
        # also a valid offset.
        _offset_lists = [self.aligned_offsets.get_offset_list(len(seq)) for seq in self.sequences]
        self.padded_offset_lists = [
            length + (self.n_offsets - len(length)) * [0] for length in _offset_lists
        ]

        # Mask used in expectation-maximization steps and log likelihood.
        self.offset_mask = tf.constant(
            [len(length) * [1] + (self.n_offsets - len(length)) * [0] for length in _offset_lists],
            dtype=self.tf_precision,
        )

        # Expand to (sequences, 1, offsets) for broadcasting to shape
        # (sequences, classes, offsets)
        self.offset_mask = tf.expand_dims(self.offset_mask, axis=1)

    def _compute_indices(self) -> None:
        """Initialize auxiliary tensors.

        These will be used by the tensorflow function 'gather_nd()' and 'scatter_nd()' within the
        EM steps.
        """
        # (sequences, classes, aligned offsets) <--> (sequences, classes, right-padded offsets)
        self.offset_weight_indices = tf.constant(
            [
                [[(c, o) for o in offset_list] for c in range(self.n_classes)]
                for offset_list in self.padded_offset_lists
            ],
            dtype=tf.int32,
        )

        # (sequences, classes, aligned offsets) <--> (classes, motif length, alphabet length)
        self.sequence_indices = tf.constant(
            [
                [
                    [
                        [
                            (c, pos, self.padded_sequences[s][i + pos])
                            for pos in range(self.motif_length)
                        ]
                        for i in range(self.n_offsets)
                    ]
                    for c in range(self.n_classes)
                ]
                for s in range(self.n_sequences)
            ],
            dtype=tf.int32,
        )

        # (sequences, classes, aligned offsets) <--> (sequences, classes, right-padded offsets)
        self.responsibility_indices = tf.constant(
            [
                [[(s, c, o) for o in offset_list] for c in range(self.n_classes)]
                for s, offset_list in enumerate(self.padded_offset_lists)
            ],
            dtype=tf.int32,
        )

    def _expectation_maximization(self) -> None:
        """Do one expectation-maximization run.

        Does one EM run until convergence and sets at least the following attributes:
        - self.current_class_weights
        - self.current_ppm
        - self.current_score
        - self.current_responsibilities
        - self.current_steps
        - self.current_pssm
        - self.current_log_likelihood_ppm
        """
        (
            class_weights,
            ppm,
            pssm,
            ll_pssm,
            ll_ppm,
            responsibilities,
            steps,
        ) = _run_expectation_maximization(
            tf.constant(self.min_error, dtype=self.tf_precision),
            tf.constant(self._initialize_responsibilities(), dtype=self.tf_precision),
            self.responsibility_indices,
            self.sequence_indices,
            self.offset_weight_indices,
            self.offset_mask,
            self.similarity_weights,
            tf.constant(self.pseudocount, dtype=self.tf_precision),
            self.background_freqs_broadcast,
            self.offset_upweighting,
        )

        self.current_class_weights = class_weights.numpy()
        self.current_ppm = ppm.numpy()
        self.current_score = float(ll_pssm.numpy())
        self.current_responsibilities = responsibilities.numpy()
        self.current_steps = int(steps.numpy())

        self.current_pssm = pssm.numpy()
        self.current_log_likelihood_ppm = float(ll_ppm.numpy())
