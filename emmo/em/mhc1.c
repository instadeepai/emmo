/* C extension for running the expectation maximization algorithm for MHC1
 * ligands. */
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <stdio.h>

/* Define a struct to hold the data for the expectation maximization algorithm.
 *
 * Throughout the code, the following variables are used to iterate over the
 * dimensions of the arrays or for similar purposes:
 *  - s: sequence index         -- iterate up to n_sequences
 *  - c: class index            -- iterate up to n_classes
 *  - p: position in the motif  -- iterate up to motif_length
 *  - a: amino acid index       -- iterate up to n_alphabet
 *  - i: index for other arrays -- e.g. iterate up to n_lengths
 */
typedef struct {
  int motif_length;
  int n_term;
  int c_term;
  int n_sequences;
  int n_classes;
  int n_alphabet;
  int n_lengths;
  int max_seq_length;
  int steps;
  int verbose;
  double min_error;
  double pseudocount;
  double n_term_penalty;
  double c_term_penalty;
  double log_likelihood;
  npy_double *ppm;
  npy_double *responsibilities;
  npy_uint16 *sequences;
  npy_uint16 *lengths;
  npy_double *class_weights;
  npy_uint16 *n_positions;
  npy_uint16 *c_positions;
} EMStruct;

static PyObject *run_em(PyObject *, PyObject *);
PyMODINIT_FUNC PyInit_mhc1_c_ext(void);
static void init_c_positions(EMStruct *);
static void print_em_data(EMStruct *);
static int expectation_maximization(EMStruct *);
static int compute_log_likelihood(EMStruct *, double *);
static int expectation(EMStruct *);
static int maximization(EMStruct *);

/* Method definitions */
static PyMethodDef Methods[] = {{"run_em", run_em, METH_VARARGS,
                                 "Run the expectation maximization algorithm"},
                                {NULL, NULL, 0, NULL}};

/* Module definition */
static struct PyModuleDef mhc1_c_ext_module = {
    PyModuleDef_HEAD_INIT, "mhc1_c_ext_module", /* name of module */
    "Module for running the expectation maximization algorithm for MHC1 "
    "ligands.",
    /* module documentation, may be NULL */
    -1, /* size of per-interpreter state of the module,
         * or -1 if the module keeps state in global variables. */
    Methods};

/* Function to run the expectation maximization algorithm (exposed by the
 * module) */
static PyObject *run_em(PyObject *self, PyObject *args) {
  /* Parse the input tuple */
  PyArrayObject *ppm, *responsibilities, *sequences, *lengths;
  int motif_length, n_term, c_term, n_lengths, verbose;
  double min_error, pseudocount, n_term_penalty, c_term_penalty;
  if (!PyArg_ParseTuple(args, "O!O!O!O!iiiiidddd", &PyArray_Type, &ppm,
                        &PyArray_Type, &responsibilities, &PyArray_Type,
                        &sequences, &PyArray_Type, &lengths, &motif_length,
                        &n_term, &c_term, &n_lengths, &verbose, &min_error,
                        &pseudocount, &n_term_penalty, &c_term_penalty)) {
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
  responsibilities = (PyArrayObject *)PyArray_FROM_OTF(
      (PyObject *)responsibilities, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
  if (responsibilities == NULL || PyArray_NDIM(responsibilities) != 2) {
    if (responsibilities != NULL) {
      PyErr_SetString(PyExc_ValueError,
                      "responsibilities must have 2 dimensions");
    }
    Py_XDECREF(ppm);
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
    Py_XDECREF(responsibilities);
    Py_XDECREF(sequences);
    Py_XDECREF(lengths);
    return NULL;
  }

  int n_sequences = PyArray_DIMS(sequences)[0];
  int n_classes = PyArray_DIMS(responsibilities)[1];
  int n_alphabet = PyArray_DIMS(ppm)[2];
  int max_seq_length = PyArray_DIMS(sequences)[1];

  npy_intp class_weight_dims[2] = {n_lengths, n_classes};
  npy_intp n_c_positions_dims[2] = {n_sequences, n_classes};

  PyArrayObject *class_weights =
      (PyArrayObject *)PyArray_ZEROS(2, class_weight_dims, NPY_DOUBLE, 0);
  if (!class_weights) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create class_weights array");
    Py_XDECREF(lengths);
    Py_XDECREF(ppm);
    Py_XDECREF(responsibilities);
    Py_XDECREF(sequences);
    return NULL;
  }
  PyArrayObject *n_positions =
      (PyArrayObject *)PyArray_ZEROS(2, n_c_positions_dims, NPY_UINT16, 0);
  if (!n_positions) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create n_positions array");
    Py_XDECREF(lengths);
    Py_XDECREF(ppm);
    Py_XDECREF(responsibilities);
    Py_XDECREF(sequences);
    Py_XDECREF(class_weights);
    return NULL;
  }
  PyArrayObject *c_positions =
      (PyArrayObject *)PyArray_SimpleNew(2, n_c_positions_dims, NPY_UINT16);
  if (!c_positions) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create c_positions array");
    Py_XDECREF(lengths);
    Py_XDECREF(ppm);
    Py_XDECREF(responsibilities);
    Py_XDECREF(sequences);
    Py_XDECREF(class_weights);
    Py_XDECREF(n_positions);
    return NULL;
  }

  EMStruct em = {motif_length,
                 n_term,
                 c_term,
                 n_sequences,
                 n_classes,
                 n_alphabet,
                 n_lengths,
                 max_seq_length,
                 0, /* steps */
                 verbose,
                 min_error,
                 pseudocount,
                 n_term_penalty,
                 c_term_penalty,
                 0.0, /* log_likelihood */
                 (npy_double *)PyArray_DATA(ppm),
                 (npy_double *)PyArray_DATA(responsibilities),
                 (npy_uint16 *)PyArray_DATA(sequences),
                 (npy_uint16 *)PyArray_DATA(lengths),
                 (npy_double *)PyArray_DATA(class_weights),
                 (npy_uint16 *)PyArray_DATA(n_positions),
                 (npy_uint16 *)PyArray_DATA(c_positions)};

  /* Initialize and run expectation maximization */
  if (verbose) {
    printf(
        "Running expectation maximization algorithm (C implementation) ...\n");
    print_em_data(&em);
  }

  init_c_positions(&em);

  if (expectation_maximization(&em) == -1) {
    Py_XDECREF(lengths);
    Py_XDECREF(ppm);
    Py_XDECREF(responsibilities);
    Py_XDECREF(sequences);
    Py_XDECREF(class_weights);
    Py_XDECREF(n_positions);
    Py_XDECREF(c_positions);
    return NULL;
  }

  /* Clean up (decrement counter for all Python objects that are not returned)
   */
  Py_XDECREF(sequences);
  Py_XDECREF(lengths);

  /* Return the results as a tuple */
  return Py_BuildValue("NNNNNdi", ppm, responsibilities, class_weights,
                       n_positions, c_positions, em.log_likelihood, em.steps);
}

/* Module initialization function */
PyMODINIT_FUNC PyInit_mhc1_c_ext(void) {
  import_array(); /* Initialize the NumPy API */
  return PyModule_Create(&mhc1_c_ext_module);
}

/* Print for debugging */
static void print_em_data(EMStruct *em) {
  printf("--- EMStruct at %p ---\n", em);

  /* integers */
  printf("motif_length = %d\n", em->motif_length);
  printf("n_term = %d\n", em->n_term);
  printf("c_term = %d\n", em->c_term);
  printf("n_sequences = %d\n", em->n_sequences);
  printf("n_classes = %d\n", em->n_classes);
  printf("n_alphabet = %d\n", em->n_alphabet);
  printf("n_lengths = %d\n", em->n_lengths);
  printf("max_seq_length = %d\n", em->max_seq_length);
  printf("steps = %d\n", em->steps);

  /* doubles */
  printf("min_error = %f\n", em->min_error);
  printf("pseudocount = %f\n", em->pseudocount);
  printf("n_term_penalty = %f\n", em->n_term_penalty);
  printf("c_term_penalty = %f\n", em->c_term_penalty);
  printf("log_likelihood = %f\n", em->log_likelihood);

  /* pointers */
  printf("address of ppm = %p\n", em->ppm);
  printf("address of responsibilities = %p\n", em->responsibilities);
  printf("address of sequences = %p\n", em->sequences);
  printf("address of lengths = %p\n", em->lengths);
  printf("address of class_weights = %p\n", em->class_weights);
  printf("address of n_positions = %p\n", em->n_positions);
  printf("address of c_positions = %p\n", em->c_positions);
  printf("------\n");
}

/* Initialize the N- and C-terminal positions */
static void init_c_positions(EMStruct *em) {
  npy_uint16 *c_positions_data = em->c_positions;
  npy_uint16 *lengths_data = em->lengths;
  int n_sequences = em->n_sequences;
  int n_classes = em->n_classes;
  int c_term = em->c_term;

  for (int s = 0; s < n_sequences; ++s) {
    npy_uint16 c_start = lengths_data[s] - c_term;
    for (int c = 0; c < n_classes; ++c) {
      c_positions_data[s * n_classes + c] = c_start;
    }
  }
}

/* Run the expectation maximization algorithm */
static int expectation_maximization(EMStruct *em) {
  double log_likelihood = 0.0, new_log_likelihood = 0.0;

  if (maximization(em) == -1) {
    return -1;
  }
  if (compute_log_likelihood(em, &log_likelihood) == -1) {
    return -1;
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
    if (compute_log_likelihood(em, &new_log_likelihood) == -1) {
      return -1;
    }

    if (log_likelihood >= new_log_likelihood) {
      log_likelihood_error = log_likelihood - new_log_likelihood;
    } else {
      log_likelihood_error = new_log_likelihood - log_likelihood;
    }
    log_likelihood = new_log_likelihood;

    if (em->verbose) {
      printf("Estimating frequencies, %d EM steps, logL = %f ...\n", steps + 1,
             log_likelihood);
    }
  }

  em->log_likelihood = log_likelihood;
  em->steps = steps;

  return 0; /* Success */
}

static npy_double class_likelihood(EMStruct *em, npy_double *class_weights_row,
                                   int s, int c, int length,
                                   int set_positions) {
  npy_double prob = class_weights_row[c];
  int seq_offset = s * em->max_seq_length;
  int class_offset = c * em->motif_length * em->n_alphabet;

  /* Case 1: If the sequence and motif length are equal, then the whole
   * sequence contributes */
  if (length == em->motif_length) {
    for (int p = 0; p < length; ++p) {
      int a = em->sequences[seq_offset + p];
      prob *= em->ppm[class_offset + p * em->n_alphabet + a];
    }
  }
  /* Case 2: If sequence length < motif length or the N-/C-positions are already
   * set (i.e. we are not in the expectation step), we only consider these fixed
   * N- and C-terminal parts */
  else if ((length < em->motif_length) || (!set_positions)) {
    npy_uint16 n_start = em->n_positions[s * em->n_classes + c];
    npy_uint16 c_start = em->c_positions[s * em->n_classes + c];
    prob *= (pow(em->n_term_penalty, n_start) *
             pow(em->c_term_penalty, length - c_start - em->c_term));

    /* Contribution of the N-terminal part of the motif */
    for (int p = n_start; p < n_start + em->n_term; ++p) {
      int a = em->sequences[seq_offset + p];
      prob *= em->ppm[class_offset + (p - n_start) * em->n_alphabet + a];
    }

    /* Contribution of the C-terminal part of the motif */
    for (int p = c_start; p < c_start + em->c_term; ++p) {
      int a = em->sequences[seq_offset + p];
      prob *= em->ppm[class_offset +
                      (em->motif_length - c_start - em->c_term + p) *
                          em->n_alphabet +
                      a];
    }
  }
  /* Case 3: If sequence length > motif length and the N-/C-positions need
   * to be set (i.e. we are in the expectation step), we need to compute
   * the maximum probability over all possible N- and C-terminal positions */
  else {
    /* prob needs to be initialized to 0 in this case because a maximum is
     * needed */
    prob = 0.0;

    for (npy_uint16 n_start = 0; n_start < length - em->motif_length + 1;
         ++n_start) {
      /* Contribution of the N-terminal part of the motif */
      npy_double current_prob_n =
          class_weights_row[c] * pow(em->n_term_penalty, n_start);
      for (int p = 0; p < em->n_term; ++p) {
        int a = em->sequences[seq_offset + p + n_start];
        current_prob_n *= em->ppm[class_offset + p * em->n_alphabet + a];
      }

      for (npy_uint16 c_start = n_start + em->motif_length - em->c_term;
           c_start < length - em->c_term + 1; ++c_start) {
        /* Contribution of the C-terminal part of the motif */
        npy_double current_prob =
            current_prob_n *
            pow(em->c_term_penalty, length - c_start - em->c_term);
        for (int p = 0; p < em->c_term; ++p) {
          int a = em->sequences[seq_offset + p + c_start];
          current_prob *=
              em->ppm[class_offset +
                      (em->motif_length - em->c_term + p) * em->n_alphabet + a];
        }

        if (current_prob > prob) {
          prob = current_prob;
          em->n_positions[s * em->n_classes + c] = n_start;
          em->c_positions[s * em->n_classes + c] = c_start;
        }
      }
    }
  }

  return prob;
}

/* Compute the log likelihood from the current parameters */
static int compute_log_likelihood(EMStruct *em, double *p_log_likelihood) {
  double log_likelihood = 0.0;
  npy_double *class_weights_row = em->class_weights;

  for (int s = 0; s < em->n_sequences; ++s) {
    npy_uint16 length = em->lengths[s];
    double prob_all_classes = 0.0;

    for (int c = 0; c < em->n_classes; ++c) {
      /* Compute the likelihood of the sequence given the current class,
       * do not set the N- and C-terminal positions */
      prob_all_classes +=
          class_likelihood(em, class_weights_row, s, c, length, 0);
    }

    if (prob_all_classes <= 0.0) {
      printf("prob_all_classes: %f\n", prob_all_classes);
      PyErr_SetString(PyExc_ValueError,
                      "Logarithm of zero or negative number encountered.");
      return -1;
    }
    log_likelihood += log(prob_all_classes);

    /* The following relies on the sequences being sorted by length */
    if ((s < em->n_sequences - 1) && (em->lengths[s] != em->lengths[s + 1])) {
      /* Move the pointer to the next row in the class_weights array */
      class_weights_row += em->n_classes;
    }
  }

  /* Add the log likelihood of the Dirichlet priors */
  double prior_log_likelihood = 0.0;
  for (int c = 0; c < em->n_classes - 1; ++c) { /* Exclude flat motif */
    int class_offset = c * em->motif_length * em->n_alphabet;
    for (int p = 0; p < em->motif_length; ++p) {
      int pos_offset = p * em->n_alphabet;
      for (int a = 0; a < em->n_alphabet; ++a) {
        double prob = em->ppm[class_offset + pos_offset + a];
        if (prob <= 0.0) {
          PyErr_SetString(PyExc_ValueError,
                          "Logarithm of zero or negative number encountered.");
          return -1;
        }
        prior_log_likelihood += log(prob);
      }
    }
  }
  log_likelihood += em->pseudocount * prior_log_likelihood;

  *p_log_likelihood = log_likelihood;
  return 0;
}

/* Expectation step: compute the responsibilities from the current PPM and class
 * weights */
static int expectation(EMStruct *em) {
  npy_double *class_weights_row = em->class_weights;
  npy_double *responsibilities_row = em->responsibilities;

  for (int s = 0; s < em->n_sequences; ++s) {
    npy_uint16 length = em->lengths[s];

    for (int c = 0; c < em->n_classes; ++c) {
      /* Compute the likelihood of the sequence given the current class,
       * also set the N- and C-terminal positions */
      responsibilities_row[c] =
          class_likelihood(em, class_weights_row, s, c, length, 1);
    }

    /* Normalize the responsibilities */
    double sum = 0.0;
    for (int c = 0; c < em->n_classes; ++c) {
      sum += responsibilities_row[c];
    }
    if (sum == 0.0) {
      PyErr_SetString(PyExc_ZeroDivisionError,
                      "Division by zero encountered in expectation step.");
      return -1;
    }
    for (int c = 0; c < em->n_classes; ++c) {
      responsibilities_row[c] /= sum;
    }

    /* Move the pointer to the next row in the responsibilities array */
    responsibilities_row += em->n_classes;

    /* The following relies on the sequences being sorted by length */
    if ((s < em->n_sequences - 1) && (em->lengths[s] != em->lengths[s + 1])) {
      /* Move the pointer to the next row in the class_weights array */
      class_weights_row += em->n_classes;
    }
  }

  return 0; /* Success */
}

/* Maximization step: update the PPM and class weights */
static int maximization(EMStruct *em) {
  /* ----------------------------------------
   * Maximization step for the PPM
   * ---------------------------------------- */

  /* Set all values to pseudocount */
  for (int c = 0; c < em->n_classes - 1; ++c) {
    int class_offset = c * em->motif_length * em->n_alphabet;
    for (int p = 0; p < em->motif_length; ++p) {
      int pos_offset = p * em->n_alphabet;
      for (int a = 0; a < em->n_alphabet; ++a) {
        em->ppm[class_offset + pos_offset + a] = em->pseudocount;
      }
    }
  }

  /* Add up responsibilities */
  for (int s = 0; s < em->n_sequences; ++s) {
    npy_uint16 length = em->lengths[s];
    int seq_offset = s * em->max_seq_length;

    for (int c = 0; c < em->n_classes - 1; ++c) {
      int class_offset = c * em->motif_length * em->n_alphabet;
      npy_double resp = em->responsibilities[s * em->n_classes + c];

      /* If the sequence and motif length are equal, then the whole sequence
       * contributes */
      if (length == em->motif_length) {
        for (int p = 0; p < length; ++p) {
          int a = em->sequences[s * em->max_seq_length + p];
          em->ppm[class_offset + p * em->n_alphabet + a] += resp;
        }
        continue;
      }

      /* Otherwise, only the N- and C-terminal positions contribute */
      npy_uint16 n_start = em->n_positions[s * em->n_classes + c];
      npy_uint16 c_start = em->c_positions[s * em->n_classes + c];

      /* Contribution of the N-terminal part of the motif */
      for (int p = 0; p < em->n_term; ++p) {
        int a = em->sequences[seq_offset + p + n_start];
        em->ppm[class_offset + p * em->n_alphabet + a] += resp;
      }

      /* Contribution of the C-terminal part of the motif */
      for (int p = 0; p < em->c_term; ++p) {
        int a = em->sequences[seq_offset + p + c_start];
        em->ppm[class_offset +
                (em->motif_length - em->c_term + p) * em->n_alphabet + a] +=
            resp;
      }
    }
  }

  /* Normalize so that frequencies sum to one for each position */
  npy_double *pos_in_ppm = em->ppm;
  for (int c = 0; c < em->n_classes - 1; ++c) {
    for (int p = 0; p < em->motif_length; ++p) {
      npy_double resp_sum = 0.0;
      for (int a = 0; a < em->n_alphabet; ++a) {
        resp_sum += pos_in_ppm[a];
      }
      if (resp_sum == 0.0) {
        PyErr_SetString(
            PyExc_ZeroDivisionError,
            "Division by zero encountered in PPM maximization step.");
        return -1;
      }
      for (int a = 0; a < em->n_alphabet; ++a) {
        pos_in_ppm[a] /= resp_sum;
      }
      /* Move the pointer*/
      pos_in_ppm += em->n_alphabet;
    }
  }

  /* ----------------------------------------
   * Maximization step for the class weights
   * ---------------------------------------- */

  /* Set all values to zero */
  npy_double *class_weights_row = em->class_weights;
  for (int i = 0; i < em->n_lengths; ++i) {
    for (int c = 0; c < em->n_classes; ++c) {
      em->class_weights[c] = 0.0;
    }
    /* Move the pointer to the next row in the class_weights array */
    class_weights_row += em->n_classes;
  }

  /* Add up the responsibilities */
  class_weights_row = em->class_weights;
  for (int s = 0; s < em->n_sequences; ++s) {
    int seq_offset = s * em->n_classes;
    for (int c = 0; c < em->n_classes; ++c) {
      class_weights_row[c] += em->responsibilities[seq_offset + c];
    }

    /* The following relies on the sequences being sorted by length */
    if ((s < em->n_sequences - 1) && (em->lengths[s] != em->lengths[s + 1])) {
      /* Move the pointer to the next row in the class_weights array */
      class_weights_row += em->n_classes;
    }
  }

  /* Normalize all rows such that they sum to 1 */
  class_weights_row = em->class_weights;
  for (int i = 0; i < em->n_lengths; ++i) {
    npy_double resp_sum = 0.0;
    for (int c = 0; c < em->n_classes; ++c) {
      resp_sum += class_weights_row[c];
    }
    if (resp_sum == 0.0) {
      PyErr_SetString(
          PyExc_ZeroDivisionError,
          "Division by zero encountered in class weights maximization step.");
      return -1;
    }
    for (int c = 0; c < em->n_classes; ++c) {
      class_weights_row[c] /= resp_sum;
    }
    /* Move the pointer to the next row in the class_weights array */
    class_weights_row += em->n_classes;
  }

  return 0; /* Success */
}
