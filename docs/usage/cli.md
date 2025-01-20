# Usage as a Command-line Tool

Currently, the following commands are available:

| Command                                  | Description                                                                           |
| ---------------------------------------- | ------------------------------------------------------------------------------------- |
| _Deconvolution_                          |                                                                                       |
| `deconvolute`                            | Run the deconvolution for MHC ligands.                                                |
| `deconvolute-per-group`                  | Run the per-group (e.g. per allele) deconvolution for MHC ligands.                    |
| `deconvolute-for-cleavage-mhc2`          | Run the deconvolution for MHC2 cleavage models.                                       |
| `plot-deconvolution-per-group`           | Plot per-group deconvolution results for MHC ligands.                                 |
| _Prediction_                             |                                                                                       |
| `predict-mhc2`                           | Run the prediction for MHC2 peptides and alleles.                                     |
| `compile-predictor-mhc2`                 | Compile an MHC2 predictor from deconvolution results.                                 |
| `predict-from-deconvolution-models-mhc2` | Run the prediction for MHC2 peptides and alleles using deconvolution models directly. |
| _Model pulling and pushing_              |                                                                                       |
| `pull-model`                             | Pull a model from the GS models' bucket.                                              |
| `push-model`                             | Push a model into the "biondeep-models" bucket on GS.                                 |

For more details and available parameters, run

```bash
emmo {command} --help
```

and check out the next documentation pages.
