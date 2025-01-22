#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "utils.h"

#include <Python.h>
#include <math.h>

/* Fill a PPM with a given value */
void fill_ppm_with_value(int n_classes, int motif_length, int n_alphabet,
                         npy_double *ppm, double value,
                         int exclude_flat_motif) {
  exclude_flat_motif = exclude_flat_motif ? 1 : 0;

  for (int c = 0; c < n_classes - exclude_flat_motif; ++c) {
    int class_offset = c * motif_length * n_alphabet;
    for (int p = 0; p < motif_length; ++p) {
      int pos_offset = p * n_alphabet;
      for (int a = 0; a < n_alphabet; ++a) {
        ppm[class_offset + pos_offset + a] = value;
      }
    }
  }
}

/* Normalize a PPM such that the frequencies sum to one for each position */
int normalize_ppm(int n_classes, int motif_length, int n_alphabet,
                  npy_double *ppm, int exclude_flat_motif) {
  exclude_flat_motif = exclude_flat_motif ? 1 : 0;

  npy_double *pos_in_ppm = ppm;
  for (int c = 0; c < n_classes - exclude_flat_motif; ++c) {
    for (int p = 0; p < motif_length; ++p) {
      npy_double resp_sum = 0.0;
      for (int a = 0; a < n_alphabet; ++a) {
        resp_sum += pos_in_ppm[a];
      }
      if (resp_sum == 0.0) {
        PyErr_SetString(
            PyExc_ZeroDivisionError,
            "Division by zero encountered in PPM maximization step.");
        return -1;
      }
      for (int a = 0; a < n_alphabet; ++a) {
        pos_in_ppm[a] /= resp_sum;
      }
      /* Move the pointer*/
      pos_in_ppm += n_alphabet;
    }
  }

  return 0; /* Success */
}

/* Compute the prior log likelihood of the Dirichlet priors */
int compute_prior_log_likelihood(int n_classes, int motif_length,
                                 int n_alphabet, npy_double *ppm,
                                 double *p_prior_log_likelihood,
                                 int exclude_flat_motif) {
  double prior_log_likelihood = 0.0;
  exclude_flat_motif = exclude_flat_motif ? 1 : 0;

  for (int c = 0; c < n_classes - exclude_flat_motif; ++c) {
    int class_offset = c * motif_length * n_alphabet;
    for (int p = 0; p < motif_length; ++p) {
      int pos_offset = p * n_alphabet;
      for (int a = 0; a < n_alphabet; ++a) {
        double prob = ppm[class_offset + pos_offset + a];
        if (prob <= 0.0) {
          PyErr_SetString(PyExc_ValueError,
                          "Logarithm of zero or negative number encountered.");
          return -1;
        }
        prior_log_likelihood += log(prob);
      }
    }
  }

  *p_prior_log_likelihood = prior_log_likelihood;

  return 0; /* Success */
}
