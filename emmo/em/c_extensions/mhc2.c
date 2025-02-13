/* C extension for running the expectation maximization algorithm for MHC2
 * ligands. */
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <stdio.h>

#include "utils.h"

/* Define a struct to hold the data for the expectation maximization algorithm.
 *
 * Throughout the code, the following variables are used to iterate over the
 * dimensions of the arrays or for similar purposes:
 *  - s: sequence index         -- iterate up to n_sequences
 *  - c: class index            -- iterate up to n_classes
 *  - p: position in the motif  -- iterate up to motif_length
 *  - a: amino acid index       -- iterate up to n_alphabet
 *  - i: index for other arrays -- e.g. iterate up to n_offsets
 */
typedef struct {
  int motif_length;
  int n_sequences;
  int n_classes;
  int n_alphabet;
  int n_lengths;
  int n_offsets;
  int max_seq_length;
  int n_cols_in_offsets;
  int steps;
  int verbose;
  double min_error;
  double pseudocount;
  double upweight_middle_offset;
  double log_likelihood_ppm;
  double log_likelihood_pssm;
  npy_double *ppm;
  npy_double *pssm;
  npy_double *background;
  npy_double *responsibilities;
  npy_uint16 *sequences;
  npy_uint16 *lengths;
  npy_double *similarity_weights;
  npy_uint16 *offsets;
  npy_double *class_weights;
} EMStruct;

static PyObject *run_em(PyObject *, PyObject *);
PyMODINIT_FUNC PyInit_mhc2_c_ext(void);
static void print_em_data(EMStruct *);
static int check_background(PyArrayObject *, int);
static int expectation_maximization(EMStruct *);
static void update_pssm(EMStruct *);
static int compute_log_likelihood(EMStruct *, double *, int);
static int expectation(EMStruct *);
static int maximization(EMStruct *);

/* Method definitions */
static PyMethodDef Methods[] = {{"run_em", run_em, METH_VARARGS,
                                 "Run the expectation maximization algorithm"},
                                {NULL, NULL, 0, NULL}};

/* Module definition */
static struct PyModuleDef mhc2_c_ext_module = {
    PyModuleDef_HEAD_INIT, "mhc2_c_ext_module", /* name of module */
    "Module for running the expectation maximization algorithm for MHC2 "
    "ligands.",
    /* module documentation, may be NULL */
    -1, /* size of per-interpreter state of the module,
         * or -1 if the module keeps state in global variables. */
    Methods};

/* Function to run the expectation maximization algorithm (exposed by the
 * module) */
static PyObject *run_em(PyObject *self, PyObject *args) {
  /* Parse the input tuple */
  PyArrayObject *ppm, *background, *responsibilities, *sequences, *lengths,
      *similarity_weights, *offsets;
  int motif_length, verbose;
  double min_error, pseudocount, upweight_middle_offset;
  if (!PyArg_ParseTuple(
          args, "O!O!O!O!O!O!O!iiddd", &PyArray_Type, &ppm, &PyArray_Type,
          &background, &PyArray_Type, &responsibilities, &PyArray_Type,
          &sequences, &PyArray_Type, &lengths, &PyArray_Type,
          &similarity_weights, &PyArray_Type, &offsets, &motif_length, &verbose,
          &min_error, &pseudocount, &upweight_middle_offset)) {
    return NULL;
  }

  /* Ensure the input arrays are contiguous arrays of the correct type. In case
   * an input array already has the correct format, the function
   * PyArray_FROM_OTF increases the reference count of the array. */
  ppm = (PyArrayObject *)PyArray_FROM_OTF((PyObject *)ppm, NPY_DOUBLE,
                                          NPY_ARRAY_C_CONTIGUOUS);
  if (ppm == NULL || PyArray_NDIM(ppm) != 3) {
    if (ppm != NULL) {
      PyErr_SetString(PyExc_ValueError, "ppm must have 3 dimensions");
      Py_XDECREF(ppm);
    }
    return NULL;
  }
  background = (PyArrayObject *)PyArray_FROM_OTF(
      (PyObject *)background, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
  if (background == NULL || PyArray_NDIM(background) != 1) {
    if (background != NULL) {
      PyErr_SetString(PyExc_ValueError, "background must have 1 dimension");
    }
    Py_XDECREF(ppm);
    Py_XDECREF(background);
    return NULL;
  }
  responsibilities = (PyArrayObject *)PyArray_FROM_OTF(
      (PyObject *)responsibilities, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
  if (responsibilities == NULL || PyArray_NDIM(responsibilities) != 3) {
    if (responsibilities != NULL) {
      PyErr_SetString(PyExc_ValueError,
                      "responsibilities must have 3 dimensions");
    }
    Py_XDECREF(ppm);
    Py_XDECREF(background);
    Py_XDECREF(responsibilities);
    return NULL;
  }
  sequences = (PyArrayObject *)PyArray_FROM_OTF(
      (PyObject *)sequences, NPY_UINT16, NPY_ARRAY_C_CONTIGUOUS);
  if (sequences == NULL || PyArray_NDIM(sequences) != 2) {
    if (sequences != NULL) {
      PyErr_SetString(PyExc_ValueError, "sequences must have 2 dimensions");
    }
    Py_XDECREF(ppm);
    Py_XDECREF(background);
    Py_XDECREF(responsibilities);
    Py_XDECREF(sequences);
    return NULL;
  }
  lengths = (PyArrayObject *)PyArray_FROM_OTF((PyObject *)lengths, NPY_UINT16,
                                              NPY_ARRAY_C_CONTIGUOUS);
  if (lengths == NULL || PyArray_NDIM(lengths) != 1) {
    if (lengths != NULL) {
      PyErr_SetString(PyExc_ValueError, "lengths must have 1 dimension");
    }
    Py_XDECREF(ppm);
    Py_XDECREF(background);
    Py_XDECREF(responsibilities);
    Py_XDECREF(sequences);
    Py_XDECREF(lengths);
    return NULL;
  }
  similarity_weights = (PyArrayObject *)PyArray_FROM_OTF(
      (PyObject *)similarity_weights, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
  if (similarity_weights == NULL || PyArray_NDIM(similarity_weights) != 1) {
    if (similarity_weights != NULL) {
      PyErr_SetString(PyExc_ValueError,
                      "similarity_weights must have 1 dimension");
    }
    Py_XDECREF(ppm);
    Py_XDECREF(background);
    Py_XDECREF(responsibilities);
    Py_XDECREF(sequences);
    Py_XDECREF(lengths);
    Py_XDECREF(similarity_weights);
    return NULL;
  }
  offsets = (PyArrayObject *)PyArray_FROM_OTF((PyObject *)offsets, NPY_UINT16,
                                              NPY_ARRAY_C_CONTIGUOUS);
  if (offsets == NULL || PyArray_NDIM(offsets) != 2) {
    if (offsets != NULL) {
      PyErr_SetString(PyExc_ValueError, "offsets must have 2 dimension");
    }
    Py_XDECREF(ppm);
    Py_XDECREF(background);
    Py_XDECREF(responsibilities);
    Py_XDECREF(sequences);
    Py_XDECREF(lengths);
    Py_XDECREF(similarity_weights);
    Py_XDECREF(offsets);
    return NULL;
  }

  int n_sequences = PyArray_DIMS(sequences)[0];
  int n_classes = PyArray_DIMS(responsibilities)[1];
  int n_offsets = PyArray_DIMS(responsibilities)[2];
  int n_alphabet = PyArray_DIMS(ppm)[2];
  int n_lengths = PyArray_DIMS(offsets)[0];
  int max_seq_length = PyArray_DIMS(sequences)[1];
  int n_cols_in_offsets = PyArray_DIMS(offsets)[1];

  if (check_background(background, n_alphabet) == -1) {
    Py_XDECREF(ppm);
    Py_XDECREF(background);
    Py_XDECREF(responsibilities);
    Py_XDECREF(sequences);
    Py_XDECREF(lengths);
    Py_XDECREF(similarity_weights);
    Py_XDECREF(offsets);
    return NULL;
  }

  npy_intp pssm_dims[3] = {n_classes, motif_length, n_alphabet};

  PyArrayObject *pssm =
      (PyArrayObject *)PyArray_ZEROS(3, pssm_dims, NPY_DOUBLE, 0);
  if (!pssm) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create pssm array");
    Py_XDECREF(ppm);
    Py_XDECREF(background);
    Py_XDECREF(responsibilities);
    Py_XDECREF(sequences);
    Py_XDECREF(lengths);
    Py_XDECREF(similarity_weights);
    Py_XDECREF(offsets);
    return NULL;
  }

  npy_intp class_weight_dims[2] = {n_classes, n_offsets};

  PyArrayObject *class_weights =
      (PyArrayObject *)PyArray_ZEROS(2, class_weight_dims, NPY_DOUBLE, 0);
  if (!class_weights) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create class_weights array");
    Py_XDECREF(ppm);
    Py_XDECREF(pssm);
    Py_XDECREF(background);
    Py_XDECREF(responsibilities);
    Py_XDECREF(sequences);
    Py_XDECREF(lengths);
    Py_XDECREF(similarity_weights);
    Py_XDECREF(offsets);
    return NULL;
  }

  EMStruct em = {motif_length,
                 n_sequences,
                 n_classes,
                 n_alphabet,
                 n_lengths,
                 n_offsets,
                 max_seq_length,
                 n_cols_in_offsets,
                 0, /* steps */
                 verbose,
                 min_error,
                 pseudocount,
                 upweight_middle_offset,
                 0.0, /* log_likelihood_ppm */
                 0.0, /* log_likelihood_pssm */
                 (npy_double *)PyArray_DATA(ppm),
                 (npy_double *)PyArray_DATA(pssm),
                 (npy_double *)PyArray_DATA(background),
                 (npy_double *)PyArray_DATA(responsibilities),
                 (npy_uint16 *)PyArray_DATA(sequences),
                 (npy_uint16 *)PyArray_DATA(lengths),
                 (npy_double *)PyArray_DATA(similarity_weights),
                 (npy_uint16 *)PyArray_DATA(offsets),
                 (npy_double *)PyArray_DATA(class_weights)};

  /* Initialize and run expectation maximization */
  if (verbose) {
    printf(
        "Running expectation maximization algorithm (C implementation) ...\n");
    print_em_data(&em);
  }

  if (expectation_maximization(&em) == -1) {
    Py_XDECREF(ppm);
    Py_XDECREF(pssm);
    Py_XDECREF(background);
    Py_XDECREF(responsibilities);
    Py_XDECREF(sequences);
    Py_XDECREF(lengths);
    Py_XDECREF(similarity_weights);
    Py_XDECREF(offsets);
    Py_XDECREF(class_weights);
    return NULL;
  }

  /* Clean up (decrement counter for all Python objects that are not returned)
   */
  Py_XDECREF(background);
  Py_XDECREF(sequences);
  Py_XDECREF(lengths);
  Py_XDECREF(similarity_weights);
  Py_XDECREF(offsets);

  /* Return the results as a tuple */
  return Py_BuildValue("NNNNddi", ppm, pssm, responsibilities, class_weights,
                       em.log_likelihood_ppm, em.log_likelihood_pssm, em.steps);
}

/* Module initialization function */
PyMODINIT_FUNC PyInit_mhc2_c_ext(void) {
  import_array(); /* Initialize the NumPy API */
  return PyModule_Create(&mhc2_c_ext_module);
}

/* Print for debugging */
static void print_em_data(EMStruct *em) {
  printf("--- EMStruct at %p ---\n", em);

  /* integers */
  printf("motif_length = %d\n", em->motif_length);
  printf("n_sequences = %d\n", em->n_sequences);
  printf("n_classes = %d\n", em->n_classes);
  printf("n_alphabet = %d\n", em->n_alphabet);
  printf("n_lengths = %d\n", em->n_lengths);
  printf("n_offsets = %d\n", em->n_offsets);
  printf("max_seq_length = %d\n", em->max_seq_length);
  printf("n_cols_in_offsets = %d\n", em->n_cols_in_offsets);
  printf("steps = %d\n", em->steps);

  /* doubles */
  printf("min_error = %f\n", em->min_error);
  printf("pseudocount = %f\n", em->pseudocount);
  printf("upweight middle offset factor = %f\n", em->upweight_middle_offset);
  printf("log_likelihood_ppm = %f\n", em->log_likelihood_ppm);
  printf("log_likelihood_pssm = %f\n", em->log_likelihood_pssm);

  /* pointers */
  printf("address of ppm = %p\n", em->ppm);
  printf("address of pssm = %p\n", em->pssm);
  printf("address of background = %p\n", em->background);
  printf("address of responsibilities = %p\n", em->responsibilities);
  printf("address of sequences = %p\n", em->sequences);
  printf("address of lengths = %p\n", em->lengths);
  printf("address of similarity_weights = %p\n", em->similarity_weights);
  printf("address of offsets = %p\n", em->offsets);
  printf("address of class_weights = %p\n", em->class_weights);
  printf("------\n");
}

/* Check that background has the right dimensions and is positive */
static int check_background(PyArrayObject *background, int n_alphabet) {
  if (PyArray_DIMS(background)[0] != n_alphabet) {
    PyErr_SetString(PyExc_ValueError, "background has the wrong dimensions");
    return -1;
  }

  for (int a = 0; a < n_alphabet; ++a) {
    if (*((npy_double *)PyArray_GETPTR1(background, a)) <= 0.0) {
      PyErr_SetString(PyExc_ValueError,
                      "background contains non-positive values");
      return -1;
    }
  }

  return 0; /* Success */
}

static void update_pssm(EMStruct *em) {
  for (int c = 0; c < em->n_classes; ++c) {
    int class_offset = c * em->motif_length * em->n_alphabet;
    for (int p = 0; p < em->motif_length; ++p) {
      int pos_offset = p * em->n_alphabet;
      for (int a = 0; a < em->n_alphabet; ++a) {
        npy_double prob = em->ppm[class_offset + pos_offset + a];
        npy_double bg = em->background[a];
        em->pssm[class_offset + pos_offset + a] = prob / bg;
      }
    }
  }
}

/* Run the expectation maximization algorithm */
static int expectation_maximization(EMStruct *em) {
  double log_likelihood_ppm = 0.0, log_likelihood_pssm = 0.0,
         new_log_likelihood_pssm = 0.0;

  if (maximization(em) == -1) {
    return -1;
  }
  if (compute_log_likelihood(em, &log_likelihood_pssm, 1) == -1) {
    return -1;
  }

  if (em->verbose) {
    printf("Estimating frequencies, initial logL = %f ...\n",
           log_likelihood_pssm);
  }

  int steps = 0;
  for (double log_likelihood_error = DBL_MAX;
       log_likelihood_error > em->min_error; ++steps) {
    if (expectation(em) == -1) {
      return -1;
    }
    if (maximization(em) == -1) {
      return -1;
    }
    if (compute_log_likelihood(em, &new_log_likelihood_pssm, 1) == -1) {
      return -1;
    }

    if (log_likelihood_pssm >= new_log_likelihood_pssm) {
      log_likelihood_error = log_likelihood_pssm - new_log_likelihood_pssm;
    } else {
      log_likelihood_error = new_log_likelihood_pssm - log_likelihood_pssm;
    }
    log_likelihood_pssm = new_log_likelihood_pssm;

    if (em->verbose) {
      printf("Estimating frequencies, %d EM steps, logL = %f ...\n", steps + 1,
             log_likelihood_pssm);
    }
  }

  /* When the EM algorithm is finished, compute also the log likelihood of the
   * PPM */
  if (compute_log_likelihood(em, &log_likelihood_ppm, 0) == -1) {
    return -1;
  }

  em->log_likelihood_pssm = log_likelihood_pssm;
  em->log_likelihood_ppm = log_likelihood_ppm;
  em->steps = steps;

  return 0; /* Success */
}

/* Compute the log likelihood from the current parameters */
static int compute_log_likelihood(EMStruct *em, double *p_log_likelihood,
                                  int use_pssm) {
  npy_double *matrix = use_pssm ? em->pssm : em->ppm;
  double log_likelihood = 0.0;
  npy_uint16 *offsets_row = em->offsets;

  for (int s = 0; s < em->n_sequences; ++s) {
    npy_uint16 length = em->lengths[s];
    npy_uint16 n_offsets_valid = length - em->motif_length + 1;
    int seq_offset = s * em->max_seq_length;
    double prob_all_classes = 0.0;

    for (int c = 0; c < em->n_classes; ++c) {
      int class_offset = c * em->motif_length * em->n_alphabet;
      for (int i = 0; i < n_offsets_valid; ++i) {
        npy_uint16 o = offsets_row[i];
        double prob = em->class_weights[c * em->n_offsets + o];
        for (int p = 0; p < em->motif_length; ++p) {
          int a = em->sequences[seq_offset + i + p];
          prob *= matrix[class_offset + p * em->n_alphabet + a];
        }
        prob_all_classes += prob;
      }
    }

    if (prob_all_classes <= 0.0) {
      PyErr_SetString(PyExc_ValueError,
                      "Logarithm of zero or negative number encountered.");
      return -1;
    }
    log_likelihood += em->similarity_weights[s] * log(prob_all_classes);

    /* The following relies on the sequences being sorted by length */
    if ((s < em->n_sequences - 1) && (em->lengths[s] != em->lengths[s + 1])) {
      /* Move the pointer to the next row in the offsets array */
      offsets_row += em->n_cols_in_offsets;
    }
  }

  /* Add the log likelihood of the Dirichlet priors */
  double prior_log_likelihood = 0.0;
  if (compute_prior_log_likelihood(em->n_classes, em->motif_length,
                                   em->n_alphabet, em->ppm,
                                   &prior_log_likelihood, 1) == -1) {
    return -1;
  }
  log_likelihood += em->pseudocount * prior_log_likelihood;

  *p_log_likelihood = log_likelihood;

  return 0; /* Success */
}

/* Expectation step: compute the responsibilities from the current PPM and class
 * weights */
static int expectation(EMStruct *em) {
  npy_uint16 *offsets_row = em->offsets;
  npy_double *responsibilities_row = em->responsibilities;

  for (int s = 0; s < em->n_sequences; ++s) {
    npy_uint16 length = em->lengths[s];
    npy_uint16 n_offsets_valid = length - em->motif_length + 1;
    int seq_offset = s * em->max_seq_length;
    double prob_all_classes = 0.0;

    for (int c = 0; c < em->n_classes; ++c) {
      int class_offset = c * em->motif_length * em->n_alphabet;
      for (int i = 0; i < n_offsets_valid; ++i) {
        npy_uint16 o = offsets_row[i];
        double prob = em->class_weights[c * em->n_offsets + o];
        for (int p = 0; p < em->motif_length; ++p) {
          int a = em->sequences[seq_offset + i + p];
          prob *= em->pssm[class_offset + p * em->n_alphabet + a];
        }
        responsibilities_row[c * em->n_offsets + o] = prob;
        prob_all_classes += prob;
      }
    }

    /* Normalize the responsibilities */
    if (prob_all_classes == 0.0) {
      PyErr_SetString(PyExc_ZeroDivisionError,
                      "Division by zero encountered in expectation step.");
      return -1;
    }
    for (int c = 0; c < em->n_classes; ++c) {
      int class_offset = c * em->n_offsets;
      for (int i = 0; i < n_offsets_valid; ++i) {
        npy_uint16 o = offsets_row[i];
        responsibilities_row[class_offset + o] /= prob_all_classes;
      }
    }

    /* Move the pointer to the next row in the responsibilities array */
    responsibilities_row += em->n_classes * em->n_offsets;

    /* The following relies on the sequences being sorted by length */
    if ((s < em->n_sequences - 1) && (em->lengths[s] != em->lengths[s + 1])) {
      /* Move the pointer to the next row in the offsets array */
      offsets_row += em->n_cols_in_offsets;
    }
  }

  return 0; /* Success */
}

/* Maximization step: update the PPM and class weights */
static int maximization(EMStruct *em) {
  /* Set all values in ppm to pseudocount */
  fill_ppm_with_value(em->n_classes, em->motif_length, em->n_alphabet, em->ppm,
                      em->pseudocount, 1);

  /* Set all values in class_weights to 0 (pseudocount is added later) */
  for (int c = 0; c < em->n_classes; ++c) {
    int class_offset = c * em->n_offsets;
    for (int i = 0; i < em->n_offsets; ++i) {
      em->class_weights[class_offset + i] = 0.0;
    }
  }

  npy_uint16 *offsets_row = em->offsets;
  npy_double *responsibilities_row = em->responsibilities;

  for (int s = 0; s < em->n_sequences; ++s) {
    npy_uint16 length = em->lengths[s];
    npy_uint16 n_offsets_valid = length - em->motif_length + 1;
    npy_double sim_weight = em->similarity_weights[s];
    int seq_offset = s * em->max_seq_length;

    for (int c = 0; c < em->n_classes; ++c) {
      int class_offset_ppm = c * em->motif_length * em->n_alphabet;
      int class_offsets_resp = c * em->n_offsets;

      for (int i = 0; i < n_offsets_valid; ++i) {
        npy_uint16 o = offsets_row[i];
        npy_double resp =
            responsibilities_row[class_offsets_resp + o] * sim_weight;

        em->class_weights[class_offsets_resp + o] += resp;

        if (c == em->n_classes - 1) { /* Flat motif */
          continue;
        }

        for (int p = 0; p < em->motif_length; ++p) {
          int a = em->sequences[seq_offset + i + p];
          em->ppm[class_offset_ppm + p * em->n_alphabet + a] += resp;
        }
      }
    }

    /* Move the pointer to the next row in the responsibilities array */
    responsibilities_row += em->n_classes * em->n_offsets;

    /* The following relies on the sequences being sorted by length */
    if ((s < em->n_sequences - 1) && (em->lengths[s] != em->lengths[s + 1])) {
      /* Move the pointer to the next row in the offsets array */
      offsets_row += em->n_cols_in_offsets;
    }
  }

  /* Normalize ppm so that frequencies sum to one for each position */
  if (normalize_ppm(em->n_classes, em->motif_length, em->n_alphabet, em->ppm,
                    1) == -1) {
    return -1;
  }

  /* Normalize class_weights so that frequencies sum to one */
  npy_double resp_sum = 0.0;
  for (int c = 0; c < em->n_classes; ++c) {
    int class_offset = c * em->n_offsets;
    for (int i = 0; i < em->n_offsets; ++i) {
      npy_double resp = em->class_weights[class_offset + i];

      /* upweight middle offset */
      if (i == em->n_offsets / 2) {
        resp *= em->upweight_middle_offset;
      }

      /* Add pseudocount */
      resp += em->pseudocount;

      resp_sum += resp;
      em->class_weights[class_offset + i] = resp;
    }
  }
  for (int c = 0; c < em->n_classes; ++c) {
    int class_offset = c * em->n_offsets;
    for (int i = 0; i < em->n_offsets; ++i) {
      em->class_weights[class_offset + i] /= resp_sum;
    }
  }

  /* Update the PSSM */
  update_pssm(em);

  return 0; /* Success */
}
