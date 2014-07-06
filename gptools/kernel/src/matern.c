/*
Copyright 2014 Stanford University and the Authors
This program is distributed under the terms of the GNU General Purpose License (GPL).
Refer to http://www.gnu.org/licenses/gpl.txt

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "matern.h"

static const double SQRT_5 = 2.2360679774997898;
static const double FIVE_THIRDS = 1.6666666666666667;

/**
 * Return the index of the first appearance of the needle in the haystack,
 * or -1 if not found
 */
static int32_t index(const int32_t *haystack, int32_t n, int32_t needle)
{
    int32_t i;
    for (i = 0; i < n; i++)
        if (haystack[i] == needle)
            return i;
    return -1;
}

/**
 * Calculate the sum of an array of integers of length n
 *
 */
static int32_t sum(const int32_t* x, int32_t n) {
    int32_t i;
    int32_t s = 0;
    for (i = 0; i < n; i++) {
        s+= x[i];
    }
    return s;
}


/**
 * Calculate the standardized squared euclidean distance between two
 * points in R^d
 *
 * \sum_{i=1}^d (X[i] - Y[i])^2 / var[i]
 */
static double seuclidean2(const double* X, const double* Y, const double* var, int32_t d)
{
    int32_t i;
    double disp;
    double r2 = 0.0;
    for (i = 0; i < d; i++) {
        disp = X[i] - Y[i];
        r2 = r2 + disp*disp / var[i];
    }
    return r2;
}


/**
 * Evaluate the direct Matern5/2 kernel between two points X, Y
 */
static double kernel(const double *X, const double *Y,
                     int32_t d, const double* var)
{

    double r2, s5r;
    r2 = seuclidean2(X, Y, var, d);
    if (r2 == 0)
        return 1;

    s5r = SQRT_5 * sqrt(r2);
    return (1.0 + s5r + FIVE_THIRDS * r2) * exp(-s5r);
}


/**
 * Evaluate the derivative of the Matern 5/2 kernel with respect
 * to the nth coordinate of X
 */
static double dkernel_dXn(const double *X, const double *Y,
                          int32_t n, int32_t d, const double* var)
{
    double r2, s5r, dr_dXn_times_r, dkernel_dr_over_r;
    r2 = seuclidean2(X, Y, var, d);
    if (r2 == 0)
        return 0;
    s5r = SQRT_5 * sqrt(r2);
    dr_dXn_times_r = (X[n] - Y[n]) / var[n];
    dkernel_dr_over_r = - FIVE_THIRDS * (1 + s5r) * exp(-s5r);
    return dkernel_dr_over_r * dr_dXn_times_r;
}

/**
 * Evaluate the 2nd derivative of the Matern 5/2 kernel with respect
 * to the nth coordinate of X and the mth coordinate of Y
 */
static double d2kernel_dXndYm(const double *X, const double *Y,
                              int32_t n, int32_t m, int32_t d,
                              const double* var)
{

    double r2, r, dr_dXn, dr_dYm, d2r_dXndYm_times_r3, s5r;
    double exp_minus_s5r, dkernel_dr_over_r, d2kernel_dr2;
    double term1, term2;

    r2 = seuclidean2(X, Y, var, d);

    if (r2 == 0) {
        if (n == m)
            return FIVE_THIRDS / (var[n]);
        return 0;
    }

    r = sqrt(r2);
    dr_dXn = (X[n] - Y[n]) / (r * var[n]);
    dr_dYm = -(X[m] - Y[m]) / (r * var[m]);

    d2r_dXndYm_times_r3 = (X[n] - Y[n]) * (X[m] - Y[m]) / (var[n]*var[m]);

    if (n == m)
        d2r_dXndYm_times_r3 -= r*r / (var[n]);

    s5r = SQRT_5 * r;
    exp_minus_s5r = exp(-s5r);
    dkernel_dr_over_r = - FIVE_THIRDS * (1 + s5r) * exp_minus_s5r;
    d2kernel_dr2 = FIVE_THIRDS * (5*r2 - s5r - 1) * exp_minus_s5r;

    term1 = dkernel_dr_over_r * d2r_dXndYm_times_r3 / r2;
    term2 = d2kernel_dr2 * dr_dXn * dr_dYm;

    return term1 + term2;
}


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
           int32_t d, const double* var)
{
    int32_t indexi = index(ni, d, 1);
    int32_t indexj = index(nj, d, 1);
    int32_t si = sum(ni, d);
    int32_t sj = sum(nj, d);
    if (si > 1 || sj > 1) {
        fprintf(stderr, "Derivative orders above 1 are not supported");
        exit(1);
    }

    if (indexi == -1 && indexj == -1)
        return kernel(xi, xj, d, var);
    if (indexi > -1 && indexj == -1)
        return dkernel_dXn(xi, xj, indexi, d, var);
    if (indexi == -1 && indexj > -1) {
        return dkernel_dXn(xj, xi, indexj, d, var);
    }
    return d2kernel_dXndYm(xi, xj, indexi, indexj, d, var);
}
