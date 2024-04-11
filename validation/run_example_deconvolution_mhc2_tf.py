"""Run example deconvolution for MHC2."""
from __future__ import annotations

import argparse

import tensorflow as tf

from emmo.constants import REPO_DIRECTORY
from emmo.em.mhc2_tf import EMRunnerMHC2
from emmo.pipeline.sequences import SequenceManager


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-n", "--no-gpu", action="store_true", help="disable GPU usage")
args = arg_parser.parse_args()

if args.no_gpu:
    print("Disabling GPU ...")
    tf.config.set_visible_devices([], "GPU")
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        if device.device_type == "GPU":
            raise RuntimeError("GPU usage could not be disabled")

input_name = "HLA-A0101_A0218_background_class_II"
directory = REPO_DIRECTORY / "validation" / "local"
file = directory / f"{input_name}.txt"
output_directory = directory / input_name

sm = SequenceManager.load_from_txt(file)
em_runner = EMRunnerMHC2(sm, 9, 2, "MHC2_biondeep", tf_precision="float64")
em_runner.run(output_directory, output_all_runs=True, force=True)
