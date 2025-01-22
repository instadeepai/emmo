#ifndef UTILS_H
#define UTILS_H

#include <numpy/arrayobject.h>

void fill_ppm_with_value(int, int, int, npy_double *, double, int);
int normalize_ppm(int, int, int, npy_double *, int);
int compute_prior_log_likelihood(int, int, int, npy_double *, double *, int);

#endif  // UTILS_H
