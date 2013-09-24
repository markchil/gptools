# Copyright 2013 Mark Chilenski
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

from .core import ArbitraryKernel

import mpmath
import scipy
import scipy.interpolate

def tanh_warp(X, l1, l2, lw, x0):
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

def spline_warp(X, l1, l2, lw, x0):
    r"""Warps the `X` coordinate with a "divot" spline shape.
    
    .. WARNING:: Broken! Do not use!
    
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
    # TODO: Why does this give non-PSD covariance matrices?
    if isinstance(X, mpmath.mpf):
        X = float(X)
    if isinstance(X, scipy.matrix):
        X = scipy.asarray(X, dtype=float).squeeze()
    return scipy.interpolate.UnivariateSpline([0.0, x0-lw/2.0, x0, x0+lw/2.0, 1.0],
                                              [l1, l1, l2, l1, l1],
                                              k=3,
                                              s=0)(X)

def gauss_warp(X, l1, l2, lw, x0):
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

class GibbsFunction1d(object):
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

class GibbsKernel1dtanh(ArbitraryKernel):
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
        super(GibbsKernel1dtanh, self).__init__(1,
                                                GibbsFunction1d(tanh_warp),
                                                **kwargs)

class GibbsKernel1dSpline(ArbitraryKernel):
    r"""Gibbs warped squared exponential covariance function in 1d.
    
    .. WARNING:: Broken! Do not use!
    
    Computes derivatives using :py:func:`mpmath.diff` and is hence in general
    much slower than a hard-coded implementation of a given kernel.
    
    The covariance function is given by
    
    .. math::

        k = \left ( \frac{2l(x)l(x')}{l^2(x)+l^2(x')} \right )^{1/2}\exp\left ( -\frac{(x-x')^2}{l^2(x)+l^2(x')} \right )
    
    Warps the length scale using a spline.
    
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
        super(GibbsKernel1dSpline, self).__init__(1,
                                                  GibbsFunction1d(spline_warp),
                                                  **kwargs)

class GibbsKernel1dGauss(ArbitraryKernel):
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
        super(GibbsKernel1dGauss, self).__init__(1,
                                                 GibbsFunction1d(gauss_warp),
                                                 **kwargs)
