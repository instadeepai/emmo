# General Notes on the Models

After having run the EM algorithm on the hits of each allele, there are 2 possibilities for
obtaining the motif in form of a position probability matrix (PPM), which are then used in the
predictor:

- the PPM output of the EM run (`direct`)
- taking all 9-mers cores estimated in the EM run and recomputing the PPM from these peptides with
  BLOSUM62-based pseudocounts (`recomputed`)

After downloading a peptide-MHC2 binding prediction model using the `emmo pull-model` command, it is
available in the folder `models/binding_predictor`.

Example usage:

```bash
emmo pull-model --model_name gs://biondeep-models/emmo/binding_predictor/mhc2_msdb_2020_full_train_tune_ppm_recomputed_20230718
```
