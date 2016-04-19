from __future__ import division

import scipy

def spev(t_int, C, deg, x, cov_C=None, M_spline=False, I_spline=False, n=0):
    """Evaluate a B-, M- or I-spline with the specified internal knots, order and coefficients.
    
    `deg` boundary knots are appended at both sides of the domain.
    
    The zeroth order basis functions are modified to ensure continuity at the
    right-hand boundary.
    
    Note that the I-splines include the :math:`i=0` case in order to have a "DC
    offset". This way your functions do not have to start at zero. If you want
    to not include this, simply set the first coefficient in `C` to zero.
    
    Parameters
    ----------
    t_int : array of float, (`M`,)
        The internal knot locations. Must be monotonic (this is NOT checked).
    C : array of float, (`M + deg - 1`,)
        The coefficients applied to the basis functions.
    deg : nonnegative int
        The polynomial degree to use.
    x : array of float, (`N`,)
        The locations to evaluate the spline at.
    cov_C : array of float, (`M + deg - 1`,) or (`M + deg - 1`, `M + deg - 1`), optional
        The covariance matrix of the coefficients. If a 1d array is passed, this
        is treated as the variance. If None, then the uncertainty is not
        computed.
    M_spline : bool, optional
        If True, compute the M-spline instead of the B-spline. M-splines are
        normalized to integrate to unity, as opposed to B-splines which sum to
        unity at all points. Default is False (compute B-spline).
    I_spline : bool, optional
        If True, compute the I-spline instead of the B-spline. Note that this
        will override `M_spline`. I-splines are the integrals of the M-splines,
        and hence ensure curves are monotonic if all coefficients are of the
        same sign. Note that the I-splines returned will be of polynomial degree
        `deg` (i.e., the integral of what is returned from calling the function
        with `deg=deg-1` and `M_spline=True`. Default is False (compute B-spline
        or M-spline).
    n : int, optional
        The derivative order to compute. Default is 0. If `n>d`, all zeros are
        returned (i.e., the discontinuities are not included).
    
    Returns
    -------
    `y` or (`y`, `cov_y`): The values (and possibly uncertainties) of the spline
    at the specified locations.
    """
    C = scipy.asarray(C, dtype=float)
    t_int = scipy.asarray(t_int, dtype=float)
    if (t_int != scipy.sort(t_int)).any():
        raise ValueError("Knots must be in increasing order!")
    # if len(scipy.unique(t_int)) != len(t_int):
    #     raise ValueError("Knots must be unique!")
    if n > deg:
        return scipy.zeros_like(x, dtype=float)
    if I_spline:
        # I_{i,k} = int_L^x M_{i,k}(u)du, so just take the derivative of the
        # underlying M-spline. Discarding the first coefficient dumps the "DC
        # offset" term.
        if cov_C is not None:
            cov_C = scipy.asarray(cov_C)
            if cov_C.ndim == 1:
                cov_C = cov_C[1:]
            elif cov_C.ndim == 2:
                cov_C = cov_C[1:, 1:]
        if n > 0:
            return spev(
                t_int, C[1:], deg - 1, x,
                cov_C=cov_C, M_spline=True, I_spline=False, n=n - 1
            )
        M_spline = True
    if n > 0:
        if M_spline:
            t = scipy.concatenate(([t_int[0]] * deg, t_int, [t_int[-1]] * deg))
            C = (deg + 1.0) * (
                C[1:] / (t[deg + 2:len(t_int) + 2 * deg] - t[1:len(t_int) + deg - 1]) -
                C[:-1] / (t[deg + 1:len(t_int) + 2 * deg - 1] - t[:len(t_int) + deg - 2])
            )
        else:
            C = C[1:] - C[:-1]
        return spev(
            t_int, C, deg - 1, x,
            cov_C=cov_C, M_spline=True, I_spline=False, n=n - 1
        )
    
    if len(C) != len(t_int) + deg - 1:
        raise ValueError("Length of C must be equal to M + deg - 1!")
    
    # Append the external knots directly at the boundary:
    t = scipy.concatenate(([t_int[0]] * deg, t_int, [t_int[-1]] * deg))
    
    # Compute the different orders:
    B = scipy.zeros((deg + 1, len(t) - 1, len(x)))
    
    # NOTE: The first dimension is indexed by deg, and is zero-indexed.
    
    # Zeroth order: constant function
    d = 0
    for i in xrange(deg, deg + len(t_int) - 2 + 1):
        # The second condition contains a hack to make the basis functions
        # continuous at the right-hand edge.
        mask = (t[i] <= x) & (
            (x < t[i + 1]) | ((i == deg + len(t_int) - 2) & (x == t[-1]))
        )
        B[d, i, mask] = 1.0 / (t[i + 1] - t[i]) if M_spline else 1.0
    
    # Loop over other orders:
    for d in xrange(1, deg + 1):
        for i in xrange(deg - d, deg + len(t_int) - 2 + 1):
            if t[i + d] != t[i]:
                v = (x - t[i]) * B[d - 1, i, :]
                if not M_spline:
                    v /= t[i + d] - t[i]
                B[d, i, :] += v
            if t[i + d + 1] != t[i + 1]:
                v = (t[i + d + 1] - x) * B[d - 1, i + 1, :]
                if not M_spline:
                    v /= t[i + d + 1] - t[i + 1]
                B[d, i, :] += v
            if M_spline and ((t[i + d] != t[i]) or (t[i + d + 1] != t[i + 1])):
                B[d, i, :] *= (d + 1) / (d * (t[i + d + 1] - t[i]))
    
    B = B[deg, 0:len(C), :].T
    
    # Now compute the I-splines, if needed:
    if I_spline:
        I = scipy.zeros_like(B)
        for i in xrange(0, len(C)):
            for m in xrange(i, len(C)):
                I[:, i] += (t[m + deg + 1] - t[m]) * B[:, m] / (deg + 1.0)
        B = I
    
    y = B.dot(C)
    if cov_C is not None:
        cov_C = scipy.asarray(cov_C)
        # If there are no covariances, promote cov_C to a diagonal matrix
        if cov_C.ndim == 1:
            cov_C = scipy.diag(cov_C)
        cov_y = B.dot(cov_C).dot(B.T)
        return (y, cov_y)
    else:
        return y
