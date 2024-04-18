# Usage as a Command-line Tool

Currently, the following commands are available:

| Command                                  | Description                                                                           |
| ---------------------------------------- | ------------------------------------------------------------------------------------- |
| _Deconvolution_                          |                                                                                       |
| `deconvolute-mhc2`                       | Run the deconvolution for MHC2 ligands.                                               |
| `deconvolute-per-allele-mhc2`            | Run the per-allele (alpha-beta pair) deconvolution for MHC2 ligands.                  |
| `deconvolute-for-cleavage-mhc2`          | Run the deconvolution for MHC2 cleavage models.                                       |
| `plot-deconvolution-per-allele-mhc2`     | Plot per-allele (alpha-beta pair) deconvolution results for MHC2 ligands.             |
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
