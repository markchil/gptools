# Copyright 2014 Mark Chilenski
# This program is distributed under the terms of the GNU General Purpose License (GPL).
# Refer to http://www.gnu.org/licenses/gpl.txt
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Provides classes and functions for creating SE kernels with warped length scales.
"""

from __future__ import division

from .core import Kernel, ArbitraryKernel
from ..utils import unique_rows

try:
    import mpmath
except ImportError:
    import warnings
    warnings.warn("Could not import mpmath. Arbitrary warping functions for Gibbs kernels will not function.",
                  ImportWarning)
import scipy
import scipy.interpolate
import inspect

def tanh_warp_arb(X, l1, l2, lw, x0):
    r"""Warps the `X` coordinate with the tanh model
    
    .. math::
    
        l = \frac{l_1 + l_2}{2} - \frac{l_1 - l_2}{2}\tanh\frac{x-x_0}{l_w}
    
    Parameters
    ----------
    X : :py:class:`Array`, (`M`,) or scalar float
        `M` locations to evaluate length scale at.
    l1 : positive float
        Small-`X` saturation value of the length scale.
    l2 : positive float
        Large-`X` saturation value of the length scale.
    lw : positive float
        Length scale of the transition between the two length scales.
    x0 : float
        Location of the center of the transition between the two length scales.
    
    Returns
    -------
    l : :py:class:`Array`, (`M`,) or scalar float
        The value of the length scale at the specified point.
    """
    if isinstance(X, scipy.ndarray):
        if isinstance(X, scipy.matrix):
            X = scipy.asarray(X, dtype=float)
        return 0.5 * ((l1 + l2) - (l1 - l2) * scipy.tanh((X - x0) / lw))
    else:
        return 0.5 * ((l1 + l2) - (l1 - l2) * mpmath.tanh((X - x0) / lw))

def gauss_warp_arb(X, l1, l2, lw, x0):
    r"""Warps the `X` coordinate with a Gaussian-shaped divot.
    
    .. math::
        
        l = l_1 - (l_1 - l_2) \exp\left ( -4\ln 2\frac{(X-x_0)^2}{l_{w}^{2}} \right )
    
    Parameters
    ----------
    X : :py:class:`Array`, (`M`,) or scalar float
        `M` locations to evaluate length scale at.
    l1 : positive float
        Global value of the length scale.
    l2 : positive float
        Pedestal value of the length scale.
    lw : positive float
        Width of the dip.
    x0 : float
        Location of the center of the dip in length scale.
    
    Returns
    -------
    l : :py:class:`Array`, (`M`,) or scalar float
        The value of the length scale at the specified point.
    """
    if isinstance(X, scipy.ndarray):
        if isinstance(X, scipy.matrix):
            X = scipy.asarray(X, dtype=float)
        return l1 - (l1 - l2) * scipy.exp(-4.0 * scipy.log(2.0) * (X - x0)**2.0 / (lw**2.0))
    else:
        return l1 - (l1 - l2) * mpmath.exp(-4.0 * mpmath.log(2.0) * (X - x0)**2.0 / (lw**2.0))

class GibbsFunction1dArb(object):
    r"""Wrapper class for the Gibbs covariance function, permits the use of arbitrary warping.
    
    The covariance function is given by
    
    .. math::

        k = \left ( \frac{2l(x)l(x')}{l^2(x)+l^2(x')} \right )^{1/2}\exp\left ( -\frac{(x-x')^2}{l^2(x)+l^2(x')} \right )
    
    Parameters
    ----------
    warp_function : callable
        The function that warps the length scale as a function of X. Must have
        the fingerprint (`Xi`, `l1`, `l2`, `lw`, `x0`).
    """
    def __init__(self, warp_function):
        self.warp_function = warp_function
    
    def __call__(self, Xi, Xj, sigmaf, l1, l2, lw, x0):
        """Evaluate the covariance function between points `Xi` and `Xj`.
        
        Parameters
        ----------
        Xi, Xj : :py:class:`Array`, :py:class:`mpf` or scalar float
            Points to evaluate covariance between. If they are :py:class:`Array`,
            :py:mod:`scipy` functions are used, otherwise :py:mod:`mpmath`
            functions are used.
        sigmaf : scalar float
            Prefactor on covariance.
        l1, l2, lw, x0 : scalar floats
            Parameters of length scale warping function, passed to
            :py:attr:`warp_function`.
        
        Returns
        -------
        k : :py:class:`Array` or :py:class:`mpf`
            Covariance between the given points.
        """
        li = self.warp_function(Xi, l1, l2, lw, x0)
        lj = self.warp_function(Xj, l1, l2, lw, x0)
        if isinstance(Xi, scipy.ndarray):
            if isinstance(Xi, scipy.matrix):
                Xi = scipy.asarray(Xi, dtype=float)
                Xj = scipy.asarray(Xj, dtype=float)
            return sigmaf**2.0 * (scipy.sqrt(2.0 * li * lj / (li**2.0 + lj**2.0)) *
                                  scipy.exp(-(Xi - Xj)**2.0 / (li**2 + lj**2)))
        else:
            return sigmaf**2.0 * (mpmath.sqrt(2.0 * li * lj / (li**2.0 + lj**2.0)) *
                                  mpmath.exp(-(Xi - Xj)**2.0 / (li**2 + lj**2)))

class GibbsKernel1dTanhArb(ArbitraryKernel):
    r"""Gibbs warped squared exponential covariance function in 1d.
    
    Computes derivatives using :py:func:`mpmath.diff` and is hence in general
    much slower than a hard-coded implementation of a given kernel.
    
    The covariance function is given by
    
    .. math::

        k = \left ( \frac{2l(x)l(x')}{l^2(x)+l^2(x')} \right )^{1/2}\exp\left ( -\frac{(x-x')^2}{l^2(x)+l^2(x')} \right )
    
    Warps the length scale using a hyperbolic tangent:
    
    .. math::
    
        l = \frac{l_1 + l_2}{2} - \frac{l_1 - l_2}{2}\tanh\frac{x-x_0}{l_w}
    
    The order of the hyperparameters is:
    
    = ====== =======================================================================
    0 sigmaf Amplitude of the covariance function
    1 l1     Small-X saturation value of the length scale.
    2 l2     Large-X saturation value of the length scale.
    3 lw     Length scale of the transition between the two length scales.
    4 x0     Location of the center of the transition between the two length scales.
    = ====== =======================================================================
    
    Parameters
    ----------
    **kwargs
        All parameters are passed to :py:class:`~gptools.kernel.core.Kernel`.
    """
    def __init__(self, **kwargs):
        super(GibbsKernel1dTanhArb, self).__init__(GibbsFunction1dArb(tanh_warp_arb),
                                                   num_dim=1,
                                                   **kwargs)

class GibbsKernel1dGaussArb(ArbitraryKernel):
    r"""Gibbs warped squared exponential covariance function in 1d.
    
    Computes derivatives using :py:func:`mpmath.diff` and is hence in general
    much slower than a hard-coded implementation of a given kernel.
    
    The covariance function is given by
    
    .. math::

        k = \left ( \frac{2l(x)l(x')}{l^2(x)+l^2(x')} \right )^{1/2}\exp\left ( -\frac{(x-x')^2}{l^2(x)+l^2(x')} \right )
    
    Warps the length scale using a gaussian:
    
    .. math::
        
        l = l_1 - (l_1 - l_2) \exp\left ( -4\ln 2\frac{(X-x_0)^2}{l_{w}^{2}} \right )

    The order of the hyperparameters is:
    
    = ====== ==================================================
    0 sigmaf Amplitude of the covariance function
    1 l1     Global value of the length scale.
    2 l2     Pedestal value of the length scale.
    3 lw     Width of the dip.
    4 x0     Location of the center of the dip in length scale.
    = ====== ==================================================

    Parameters
    ----------
    **kwargs
        All parameters are passed to :py:class:`~gptools.kernel.core.Kernel`.
    """
    def __init__(self, **kwargs):
        super(GibbsKernel1dGaussArb, self).__init__(GibbsFunction1dArb(gauss_warp_arb),
                                                    num_dim=1,
                                                    **kwargs)

class GibbsKernel1d(Kernel):
    r"""Univariate Gibbs kernel with arbitrary length scale warping for low derivatives.
    
    The covariance function is given by
    
    .. math::
    
        k = \left ( \frac{2l(x)l(x')}{l^2(x)+l^2(x')} \right )^{1/2}\exp\left ( -\frac{(x-x')^2}{l^2(x)+l^2(x')} \right )
    
    The derivatives are hard-coded using expressions obtained from Mathematica.
    
    Parameters
    ----------
    l_func : callable
        Function that dictates the length scale warping and its derivative.
        Must have fingerprint (`x`, `n`, `p1`, `p2`, ...) where `p1` is element
        one of the kernel's parameters (i.e., element zero is skipped).
    num_params : int, optional
        The number of parameters of the length scale function. If not passed,
        introspection will be used to determine this. This will fail if you have
        used the \*args syntax in your function definition. This count should
        include sigma_f, even though it is not passed to the length scale
        function.
    **kwargs
        All remaining arguments are passed to :py:class:`~gptools.kernel.core.Kernel`.
    """
    def __init__(self, l_func, num_params=None, **kwargs):
        self.l_func = l_func
        # TODO: Some of the logic from warping should be ported over.
        if num_params is None:
            # There are two unimportant parameters at the start of the l_func
            # fingerprint, then we have to add one for sigma_f.
            try:
                num_params = len(inspect.getargspec(l_func)[0]) - 2 + 1
            except TypeError:
                # Need to remove self from the arg list for bound method:
                num_params = len(inspect.getargspec(l_func.__call__)[0]) - 3 + 1
        
        super(GibbsKernel1d, self).__init__(num_dim=1, num_params=num_params, **kwargs)
    
    def __call__(self, Xi, Xj, ni, nj, hyper_deriv=None, symmetric=False):
        """Evaluate the covariance between points `Xi` and `Xj` with derivative order `ni`, `nj`.
        
        Parameters
        ----------
        Xi : :py:class:`Matrix` or other Array-like, (`M`, `D`)
            `M` inputs with dimension `D`.
        Xj : :py:class:`Matrix` or other Array-like, (`M`, `D`)
            `M` inputs with dimension `D`.
        ni : :py:class:`Matrix` or other Array-like, (`M`, `D`)
            `M` derivative orders for set `i`.
        nj : :py:class:`Matrix` or other Array-like, (`M`, `D`)
            `M` derivative orders for set `j`.
        hyper_deriv : Non-negative int or None, optional
            The index of the hyperparameter to compute the first derivative
            with respect to. If None, no derivatives are taken. Hyperparameter
            derivatives are not supported at this point. Default is None.
        symmetric : bool, optional
            Whether or not the input `Xi`, `Xj` are from a symmetric matrix.
            Default is False.
        
        Returns
        -------
        Kij : :py:class:`Array`, (`M`,)
            Covariances for each of the `M` `Xi`, `Xj` pairs.
        
        Raises
        ------
        NotImplementedError
            If the `hyper_deriv` keyword is not None.
        """
        if hyper_deriv is not None:
            raise NotImplementedError("Hyperparameter derivatives have not been implemented!")
        
        n_combined = scipy.asarray(scipy.hstack((ni, nj)), dtype=int)
        n_combined_unique = unique_rows(n_combined)
        
        x = scipy.asarray(Xi, dtype=float)
        y = scipy.asarray(Xj, dtype=float)
        
        lx = self.l_func(x, 0, *self.params[1:])
        ly = self.l_func(y, 0, *self.params[1:])
        lx1 = self.l_func(x, 1, *self.params[1:])
        ly1 = self.l_func(y, 1, *self.params[1:])
        
        x_y = x - y
        lx2ly2 = lx**2 + ly**2
        
        k = scipy.zeros(Xi.shape[0], dtype=float)
        for n_combined_state in n_combined_unique:
            idxs = (n_combined == n_combined_state).all(axis=1)
            # Derviative expressions evaluated with Mathematica, assuming l>0.
            if (n_combined_state == scipy.asarray([0, 0])).all():
                k[idxs] = (scipy.sqrt(2.0 * lx[idxs] * ly[idxs] / lx2ly2[idxs]) *
                           scipy.exp(-x_y[idxs]**2 / lx2ly2[idxs]))
            elif (n_combined_state == scipy.asarray([1, 0])).all():
                k[idxs] = (
                    (
                        scipy.exp(-(x_y[idxs]**2 / lx2ly2[idxs])) *
                        ly[idxs] * (
                            -4 * x_y[idxs] * lx[idxs]**3 -
                            4 * x_y[idxs] * lx[idxs] * ly[idxs]**2 +
                            4 * x_y[idxs]**2 * lx[idxs]**2 * lx1[idxs] - 
                            lx[idxs]**4 * lx1[idxs] +
                            ly[idxs]**4 * lx1[idxs]
                        )
                    ) / (scipy.sqrt(2 * (lx[idxs] * ly[idxs]) / lx2ly2[idxs]) * lx2ly2[idxs]**3)
                )
            elif (n_combined_state == scipy.asarray([0, 1])).all():
                k[idxs] = (
                    (
                        scipy.exp(-(x_y[idxs]**2 / lx2ly2[idxs])) *
                        lx[idxs] * (
                            4 * x_y[idxs] * ly[idxs]**3 +
                            4 * x_y[idxs] * ly[idxs] * lx[idxs]**2 +
                            4 * x_y[idxs]**2 * ly[idxs]**2 * ly1[idxs] - 
                            ly[idxs]**4 * ly1[idxs] +
                            lx[idxs]**4 * ly1[idxs]
                        )
                    ) / (scipy.sqrt(2 * (lx[idxs] * ly[idxs]) / lx2ly2[idxs]) * lx2ly2[idxs]**3)
                )
            elif (n_combined_state == scipy.asarray([1, 1])).all():
                k[idxs] = (
                    (
                        scipy.exp(-(x_y[idxs]**2 / lx2ly2[idxs])) *
                        (
                            -lx[idxs]**8 * lx1[idxs] * ly1[idxs] +
                            4 * lx[idxs]**7 * (2 * ly[idxs] - x_y[idxs] * ly1[idxs]) -
                            4 * lx[idxs]**5 * ly[idxs] * (
                                4 * x_y[idxs]**2 -
                                6 * ly[idxs]**2 -
                                3 * x_y[idxs] * ly[idxs] * ly1[idxs]
                            ) + ly[idxs]**6 * lx1[idxs] * (
                                4 * x_y[idxs] * ly[idxs] +
                                4 * x_y[idxs]**2 * ly1[idxs] -
                                ly[idxs]**2 * ly1[idxs]
                            ) + 4 * lx[idxs]**6 * lx1[idxs] * (
                                -5 * x_y[idxs] * ly[idxs] +
                                x_y[idxs]**2 * ly1[idxs] +
                                2 * ly[idxs]**2 * ly1[idxs]
                            ) - 4 * lx[idxs] * ly[idxs]**4 * (
                                4 * x_y[idxs]**2 * ly[idxs] -
                                2 * ly[idxs]**3 +
                                4 * x_y[idxs]**3 * ly1[idxs] -
                                5 * x_y[idxs] * ly[idxs]**2 * ly1[idxs]
                            ) - 4 * lx[idxs]**3 * ly[idxs]**2 * (
                                8 * x_y[idxs]**2 * ly[idxs] -
                                6 * ly[idxs]**3 +
                                4 * x_y[idxs]**3 * ly1[idxs] -
                                9 * x_y[idxs] * ly[idxs]**2 * ly1[idxs]
                            ) + 2 * lx[idxs]**4 * ly[idxs] * lx1[idxs] * (
                                8 * x_y[idxs]**3 -
                                18 * x_y[idxs] * ly[idxs]**2 -
                                18 * x_y[idxs]**2 * ly[idxs] * ly1[idxs] +
                                9 * ly[idxs]**3 * ly1[idxs]
                            ) + 4 * lx[idxs]**2 * ly[idxs]**2 * lx1[idxs] * (
                                4 * x_y[idxs]**3 * ly[idxs] -
                                3 * x_y[idxs] * ly[idxs]**3 +
                                4 * x_y[idxs]**4 * ly1[idxs] -
                                9 * x_y[idxs]**2 * ly[idxs]**2 * ly1[idxs] +
                                2 * ly[idxs]**4 * ly1[idxs]
                            )
                        )
                    ) / (2 * scipy.sqrt(2 * (lx[idxs] * ly[idxs]) / (lx[idxs]**2 + ly[idxs]**2)) * lx2ly2[idxs]**5)
                )
            else:
                raise NotImplementedError("Derivatives greater than [1, 1] are not supported!")
        k = self.params[0]**2 * k
        return k

def tanh_warp(x, n, l1, l2, lw, x0):
    r"""Implements a tanh warping function and its derivative.
    
    .. math::
    
        l = \frac{l_1 + l_2}{2} - \frac{l_1 - l_2}{2}\tanh\frac{x-x_0}{l_w}
    
    Parameters
    ----------
    x : float or array of float
        Locations to evaluate the function at.
    n : int
        Derivative order to take. Used for ALL of the points.
    l1 : positive float
        Left saturation value.
    l2 : positive float
        Right saturation value.
    lw : positive float
        Transition width.
    x0 : float
        Transition location.

    Returns
    -------
    l : float or array
        Warped length scale at the given locations.

    Raises
    ------
    NotImplementedError
        If `n` > 1.
    """
    if n == 0:
        return (l1 + l2) / 2.0 - (l1 - l2) / 2.0 * scipy.tanh((x - x0) / lw)
    elif n == 1:
        return -(l1 - l2) / (2.0 * lw) * (scipy.cosh((x - x0) / lw))**(-2.0)
    else:
        raise NotImplementedError("Only derivatives up to order 1 are supported!")

class GibbsKernel1dTanh(GibbsKernel1d):
    r"""Gibbs warped squared exponential covariance function in 1d.
    
    Uses hard-coded implementation up to first derivatives.
    
    The covariance function is given by
    
    .. math::

        k = \left ( \frac{2l(x)l(x')}{l^2(x)+l^2(x')} \right )^{1/2}\exp\left ( -\frac{(x-x')^2}{l^2(x)+l^2(x')} \right )
    
    Warps the length scale using a hyperbolic tangent:
    
    .. math::
    
        l = \frac{l_1 + l_2}{2} - \frac{l_1 - l_2}{2}\tanh\frac{x-x_0}{l_w}
    
    The order of the hyperparameters is:
    
    = ====== =======================================================================
    0 sigmaf Amplitude of the covariance function
    1 l1     Small-X saturation value of the length scale.
    2 l2     Large-X saturation value of the length scale.
    3 lw     Length scale of the transition between the two length scales.
    4 x0     Location of the center of the transition between the two length scales.
    = ====== =======================================================================
    
    Parameters
    ----------
    **kwargs
        All parameters are passed to :py:class:`~gptools.kernel.core.Kernel`.
    """
    def __init__(self, **kwargs):
        super(GibbsKernel1dTanh, self).__init__(tanh_warp,
                                                param_names=[r'\sigma_f',
                                                             'l_1',
                                                             'l_2',
                                                             'l_w',
                                                             'x_0'],
                                                **kwargs)

def double_tanh_warp(x, n, lcore, lmid, ledge, la, lb, xa, xb):
    r"""Implements a sum-of-tanh warping function and its derivative.
    
    .. math::
    
        l = a\tanh\frac{x-x_a}{l_a} + b\tanh\frac{x-x_b}{l_b}
    
    Parameters
    ----------
    x : float or array of float
        Locations to evaluate the function at.
    n : int
        Derivative order to take. Used for ALL of the points.
    lcore : float
        Core length scale.
    lmid : float
        Intermediate length scale.
    ledge : float
        Edge length scale.
    la : positive float
        Transition of first tanh.
    lb : positive float
        Transition of second tanh.
    xa : float
        Transition of first tanh.
    xb : float
        Transition of second tanh.
    
    Returns
    -------
    l : float or array
        Warped length scale at the given locations.

    Raises
    ------
    NotImplementedError
        If `n` > 1.
    """
    a, b, c = scipy.dot([[-0.5, 0, 0.5], [0, 0.5, -0.5], [0.5, 0.5, 0]],
                        [[lcore], [ledge], [lmid]])
    a = a[0]
    b = b[0]
    c = c[0]
    if n == 0:
        return a * scipy.tanh((x - xa) / la) + b * scipy.tanh((x - xb) / lb) + c
    elif n == 1:
        return (a / la * (scipy.cosh((x - xa) / la))**(-2.0) +
                b / lb * (scipy.cosh((x - xb) / lb))**(-2.0))
    else:
        raise NotImplementedError("Only derivatives up to order 1 are supported!")

class GibbsKernel1dDoubleTanh(GibbsKernel1d):
    r"""Gibbs warped squared exponential covariance function in 1d.
    
    Uses hard-coded implementation up to first derivatives.
    
    The covariance function is given by
    
    .. math::

        k = \left ( \frac{2l(x)l(x')}{l^2(x)+l^2(x')} \right )^{1/2}\exp\left ( -\frac{(x-x')^2}{l^2(x)+l^2(x')} \right )
    
    Warps the length scale using two hyperbolic tangents:
    
    .. math::
    
        l = a\tanh\frac{x-x_a}{l_a} + b\tanh\frac{x-x_b}{l_b}
    
    The order of the hyperparameters is:
    
    = ====== ====================================
    0 sigmaf Amplitude of the covariance function
    1 lcore  Core length scale
    2 lmid   Intermediate length scale
    3 ledge  Edge length scale
    4 la     Width of first tanh
    5 lb     Width of second tanh
    6 xa     Center of first tanh
    7 xb     Center of second tanh
    = ====== ====================================
    
    Parameters
    ----------
    **kwargs
        All parameters are passed to :py:class:`~gptools.kernel.core.Kernel`.
    """
    def __init__(self, **kwargs):
        super(GibbsKernel1dDoubleTanh, self).__init__(
            double_tanh_warp,
            param_names=[r'\sigma_f', 'l_c', 'l_m', 'l_e', 'l_a', 'l_b', 'x_a', 'x_b'],
            **kwargs
        )

def cubic_bucket_warp(x, n, l1, l2, l3, x0, w1, w2, w3):
    """Warps the length scale with a piecewise cubic "bucket" shape.
    
    Parameters
    ----------
    x : float or array-like of float
        Locations to evaluate length scale at.
    n : non-negative int
        Derivative order to evaluate. Only first derivatives are supported.
    l1 : positive float
        Length scale to the left of the bucket.
    l2 : positive float
        Length scale in the bucket.
    l3 : positive float
        Length scale to the right of the bucket.
    x0 : float
        Location of the center of the bucket.
    w1 : positive float
        Width of the left side cubic section.
    w2 : positive float
        Width of the bucket.
    w3 : positive float
        Width of the right side cubic section.
    """
    x1 = x0 - w2 / 2.0 - w1 / 2.0
    x2 = x0 + w2 / 2.0 + w3 / 2.0
    x_shift_1 = (x - x1 + w1 / 2.0) / w1
    x_shift_2 = (x - x2 + w3 / 2.0) / w3
    if n == 0:
        return (
            l1 * (x <= (x1 - w1 / 2.0)) + (
                -2.0 * (l2 - l1) * (x_shift_1**3 - 3.0 / 2.0 * x_shift_1**2) + l1
            ) * ((x > (x1 - w1 / 2.0)) & (x < (x1 + w1 / 2.0))) +
            l2 * ((x >= (x1 + w1 / 2.0)) & (x <= x2 - w3 / 2.0)) + (
                -2.0 * (l3 - l2) * (x_shift_2**3 - 3.0 / 2.0 * x_shift_2**2) + l2
            ) * ((x > (x2 - w3 / 2.0)) & (x < (x2 + w3 / 2.0))) +
            l3 * (x >= (x2 + w3 / 2.0))
        )
    elif n == 1:
        return (
            (
                -2.0 * (l2 - l1) * (3 * x_shift_1**2 - 3.0 * x_shift_1) / w1
            ) * ((x > (x1 - w1 / 2.0)) & (x < (x1 + w1 / 2.0))) +
            (
                -2.0 * (l3 - l2) * (3 * x_shift_2**2 - 3.0 * x_shift_2) / w3
            ) * ((x > (x2 - w3 / 2.0)) & (x < (x2 + w3 / 2.0)))
        )
    else:
        raise NotImplementedError("Only up to first derivatives are supported!")

class GibbsKernel1dCubicBucket(GibbsKernel1d):
    r"""Gibbs warped squared exponential covariance function in 1d.

    Uses hard-coded implementation up to first derivatives.

    The covariance function is given by

    .. math::

        k = \left ( \frac{2l(x)l(x')}{l^2(x)+l^2(x')} \right )^{1/2}\exp\left ( -\frac{(x-x')^2}{l^2(x)+l^2(x')} \right )

    Warps the length scale using a "bucket" function with cubic joins.
    
    The order of the hyperparameters is:

    = ====== ========================================
    0 sigmaf Amplitude of the covariance function
    1 l1     Length scale to the left of the bucket.
    2 l2     Length scale in the bucket.
    3 l3     Length scale to the right of the bucket.
    4 x0     Location of the center of the bucket.
    5 w1     Width of the left side cubic section.
    6 w2     Width of the bucket.
    7 w3     Width of the right side cubic section.
    = ====== ========================================

    Parameters
    ----------
    **kwargs
        All parameters are passed to :py:class:`~gptools.kernel.core.Kernel`.
    """
    def __init__(self, **kwargs):
        super(GibbsKernel1dBucket, self).__init__(cubic_bucket_warp,
                                                  [r'\sigma_f',
                                                   'l_1',
                                                   'l_2',
                                                   'l_3',
                                                   'x_0',
                                                   'w_1',
                                                   'w_2',
                                                   'w_3'],
                                                  **kwargs)

def quintic_bucket_warp(x, n, l1, l2, l3, x0, w1, w2, w3):
    """Warps the length scale with a piecewise quintic "bucket" shape.

    Parameters
    ----------
    x : float or array-like of float
        Locations to evaluate length scale at.
    n : non-negative int
        Derivative order to evaluate. Only first derivatives are supported.
    l1 : positive float
        Length scale to the left of the bucket.
    l2 : positive float
        Length scale in the bucket.
    l3 : positive float
        Length scale to the right of the bucket.
    x0 : float
        Location of the center of the bucket.
    w1 : positive float
        Width of the left side quintic section.
    w2 : positive float
        Width of the bucket.
    w3 : positive float
        Width of the right side quintic section.
    """
    x1 = x0 - w2 / 2.0 - w1 / 2.0
    x2 = x0 + w2 / 2.0 + w3 / 2.0
    x_shift_1 = 2.0 * (x - x1) / w1
    x_shift_3 = 2.0 * (x - x2) / w3
    if n == 0:
        return (
            l1 * (x <= (x1 - w1 / 2.0)) + (
                0.5 * (l2 - l1) * (
                    3.0 / 8.0 * x_shift_1**5 -
                    5.0 / 4.0 * x_shift_1**3 +
                    15.0 / 8.0 * x_shift_1
                ) + (l1 + l2) / 2.0
            ) * ((x > (x1 - w1 / 2.0)) & (x < (x1 + w1 / 2.0))) +
            l2 * ((x >= (x1 + w1 / 2.0)) & (x <= x2 - w3 / 2.0)) + (
                0.5 * (l3 - l2) * (
                    3.0 / 8.0 * x_shift_3**5 -
                    5.0 / 4.0 * x_shift_3**3 +
                    15.0 / 8.0 * x_shift_3
                ) + (l2 + l3) / 2.0
            ) * ((x > (x2 - w3 / 2.0)) & (x < (x2 + w3 / 2.0))) +
            l3 * (x >= (x2 + w3 / 2.0))
        )
    elif n == 1:
        return (
            (
                0.5 * (l2 - l1) * (
                    5.0 * 3.0 / 8.0 * x_shift_1**4 -
                    3.0 * 5.0 / 4.0 * x_shift_1**2 +
                    15.0 / 8.0
                ) / w1
            ) * ((x > (x1 - w1 / 2.0)) & (x < (x1 + w1 / 2.0))) + (
                0.5 * (l3 - l2) * (
                    5.0 * 3.0 / 8.0 * x_shift_3**4 -
                    3.0 * 5.0 / 4.0 * x_shift_3**2 +
                    15.0 / 8.0
                ) / w3
            ) * ((x > (x2 - w3 / 2.0)) & (x < (x2 + w3 / 2.0)))
        )
    else:
        raise NotImplementedError("Only up to first derivatives are supported!")

class GibbsKernel1dQuinticBucket(GibbsKernel1d):
    r"""Gibbs warped squared exponential covariance function in 1d.

    Uses hard-coded implementation up to first derivatives.

    The covariance function is given by

    .. math::

        k = \left ( \frac{2l(x)l(x')}{l^2(x)+l^2(x')} \right )^{1/2}\exp\left ( -\frac{(x-x')^2}{l^2(x)+l^2(x')} \right )

    Warps the length scale using a "bucket" function with quintic joins.

    The order of the hyperparameters is:

    = ====== ========================================
    0 sigmaf Amplitude of the covariance function
    1 l1     Length scale to the left of the bucket.
    2 l2     Length scale in the bucket.
    3 l3     Length scale to the right of the bucket.
    4 x0     Location of the center of the bucket.
    5 w1     Width of the left side quintic section.
    6 w2     Width of the bucket.
    7 w3     Width of the right side quintic section.
    = ====== ========================================

    Parameters
    ----------
    **kwargs
        All parameters are passed to :py:class:`~gptools.kernel.core.Kernel`.
    """
    def __init__(self, **kwargs):
        super(GibbsKernel1dQuinticBucket, self).__init__(quintic_bucket_warp,
                                                         [r'\sigma_f',
                                                          'l_1',
                                                          'l_2',
                                                          'l_3',
                                                          'x_0',
                                                          'w_1',
                                                          'w_2',
                                                          'w_3'],
                                                         **kwargs)
