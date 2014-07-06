#ifndef MATERN_KERNEL_H_
#define MATERN_KERNEL_H_
#ifdef _MSC_VER
typedef __int32 int32_t;
#else
#include <stdint.h>
#endif

/**
 * Evaluate the Matern 5/2 kernel between a pair of points xi, xj in R^d.
 *
 * Parameters
 * ----------
 * xi : array, (d,)
 * xj : array, (d,)
 * ni : array, (d,)
 *   The derivative orders with respect to xi
 * nj : array, (d,)
 *   The derivative orders with respect to xj
 * d : int
 * var : array, (d,)
 *   The squared length scale (variance) in each direction
 */
double matern52(const double *xi, const double *xj,
                const int32_t* ni, const int32_t* nj,
                int32_t d, const double* ls);

#endif
