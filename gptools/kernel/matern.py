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

"""Provides the :py:class:`MaternKernel` class which implements the anisotropic Matern kernel.
"""

from __future__ import division

import warnings

from .core import ChainRuleKernel, ArbitraryKernel, Kernel
from ..utils import generate_set_partitions, unique_rows, yn2Kn2Der, fixed_poch
try:
    from ._matern import _matern52
except ImportError:
    warnings.warn(
        "Could not import _matern, the extension might not have been built "
        "properly. The optimized Matern52Kernel will not be available.",
        ImportWarning
    )

import scipy
import scipy.special
try:
    import mpmath
except ImportError:
    warnings.warn("Could not import mpmath. Certain functions of the Matern kernel will not function.",
                  ImportWarning)

def matern_function(Xi, Xj, *args):
    r"""Matern covariance function of arbitrary dimension, for use with :py:class:`ArbitraryKernel`.
    
    The Matern kernel has the following hyperparameters, always referenced in
    the order listed:
    
    = ===== ====================================
    0 sigma prefactor
    1 nu    order of kernel
    2 l1    length scale for the first dimension
    3 l2    ...and so on for all dimensions
    = ===== ====================================
    
    The kernel is defined as:
    
    .. math::
    
        k_M = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)}
        \left (\sqrt{2\nu \sum_i\left (\frac{\tau_i^2}{l_i^2}\right )}\right )^\nu
        K_\nu\left(\sqrt{2\nu \sum_i\left(\frac{\tau_i^2}{l_i^2}\right)}\right)
    
    Parameters
    ----------
    Xi, Xj : :py:class:`Array`, :py:class:`mpf`, tuple or scalar float
        Points to evaluate the covariance between. If they are :py:class:`Array`,
        :py:mod:`scipy` functions are used, otherwise :py:mod:`mpmath`
        functions are used.
    *args
        Remaining arguments are the 2+num_dim hyperparameters as defined above.
    """
    num_dim = len(args) - 2
    nu = args[1]
    
    if isinstance(Xi, scipy.ndarray):
        if isinstance(Xi, scipy.matrix):
            Xi = scipy.asarray(Xi, dtype=float)
            Xj = scipy.asarray(Xj, dtype=float)
        
        tau = scipy.asarray(Xi - Xj, dtype=float)
        l_mat = scipy.tile(args[-num_dim:], (tau.shape[0], 1))
        r2l2 = scipy.sum((tau / l_mat)**2, axis=1)
        y = scipy.sqrt(2.0 * nu * r2l2)
        k = 2.0**(1 - nu) / scipy.special.gamma(nu) * y**nu * scipy.special.kv(nu, y)
        k[r2l2 == 0] = 1
    else:
        try:
            tau = [xi - xj for xi, xj in zip(Xi, Xj)]
        except TypeError:
            tau = Xi - Xj
        try:
            r2l2 = sum([(t / l)**2 for t, l in zip(tau, args[2:])])
        except TypeError:
            r2l2 = (tau / args[2])**2
        y = mpmath.sqrt(2.0 * nu * r2l2)
        k = 2.0**(1 - nu) / mpmath.gamma(nu) * y**nu * mpmath.besselk(nu, y)
    k *= args[0]**2.0
    return k

class MaternKernelArb(ArbitraryKernel):
    r"""Matern covariance kernel. Supports arbitrary derivatives. Treats order as a hyperparameter.
    
    This version of the Matern kernel is painfully slow, but uses :py:mod:`mpmath`
    to ensure the derivatives are computed properly, since there may be issues
    with the regular :py:class:`MaternKernel`.
    
    The Matern kernel has the following hyperparameters, always referenced in
    the order listed:
    
    = ===== ====================================
    0 sigma prefactor
    1 nu    order of kernel
    2 l1    length scale for the first dimension
    3 l2    ...and so on for all dimensions
    = ===== ====================================
    
    The kernel is defined as:
    
    .. math::
    
        k_M = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)}
        \left (\sqrt{2\nu \sum_i\left (\frac{\tau_i^2}{l_i^2}\right )}\right )^\nu
        K_\nu\left(\sqrt{2\nu \sum_i\left(\frac{\tau_i^2}{l_i^2}\right)}\right)

    Parameters
    ----------
    **kwargs
        All keyword parameters are passed to :py:class:`~gptools.kernel.core.ArbitraryKernel`.
    """
    def __init__(self, **kwargs):
        param_names = [r'\sigma_f', r'\nu'] + ['l_%d' % (i + 1,) for i in range(0, kwargs.get('num_dim', 1))]
        super(MaternKernelArb, self).__init__(matern_function,
                                              num_params=2 + kwargs.get('num_dim', 1),
                                              param_names=param_names,
                                              **kwargs)
    
    @property
    def nu(self):
        r"""Returns the value of the order :math:`\nu`.
        """
        return self.params[1]

class MaternKernel1d(Kernel):
    r"""Matern covariance kernel. Only for univariate data. Only supports up to first derivatives. Treats order as a hyperparameter.
    
    The Matern kernel has the following hyperparameters, always referenced in
    the order listed:
    
    = ===== ===============
    0 sigma prefactor
    1 nu    order of kernel
    2 l1    length scale
    = ===== ===============
    
    The kernel is defined as:
    
    .. math::
    
        k_M = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)}
        \left (\sqrt{2\nu \left (\frac{\tau^2}{l_1^2}\right )}\right )^\nu
        K_\nu\left(\sqrt{2\nu \left(\frac{\tau^2}{l_1^2}\right)}\right)
    
    where :math:`\tau=X_i-X_j`.
    
    Note that the expressions for the derivatives break down for :math:`\nu < 1`.

    Parameters
    ----------
    **kwargs
        All keyword parameters are passed to :py:class:`~gptools.kernel.core.Kernel`.
    """
    def __init__(self, **kwargs):
        param_names = [r'\sigma_f', r'\nu', r'l_1']
        super(MaternKernel1d, self).__init__(
            num_dim=1,
            num_params=3,
            param_names=param_names,
            **kwargs
        )
    
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
        
        x_y = scipy.asarray(Xi, dtype=float).ravel() - scipy.asarray(Xj, dtype=float).ravel()
        
        zero_mask = x_y == 0
        
        q = scipy.sqrt(2 * self.params[1] * x_y**2) / self.params[2]
        
        k = scipy.zeros(Xi.shape[0], dtype=float)
        for n_combined_state in n_combined_unique:
            idxs = (n_combined == n_combined_state).all(axis=1)
            # Derviative expressions evaluated with Mathematica, assuming l>0.
            if (n_combined_state == scipy.asarray([0, 0])).all():
                k[idxs] = q[idxs]**self.params[1] * scipy.special.kv(self.params[1], q[idxs])
                k[idxs & zero_mask] = 2**(self.params[1] - 1) * scipy.special.gamma(self.params[1])
            elif (n_combined_state == scipy.asarray([1, 0])).all():
                k[idxs] = -1.0 / x_y[idxs] * q[idxs]**(1 + self.params[1]) * scipy.special.kv(self.params[1] - 1, q[idxs])
                k[idxs & zero_mask] = 0.0
            elif (n_combined_state == scipy.asarray([0, 1])).all():
                k[idxs] = 1.0 / x_y[idxs] * q[idxs]**(1 + self.params[1]) * scipy.special.kv(self.params[1] - 1, q[idxs])
                k[idxs & zero_mask] = 0.0
            elif (n_combined_state == scipy.asarray([1, 1])).all():
                k[idxs] = 2.0 * self.params[1] / (self.params[2])**2 * q[idxs]**self.params[1] * (
                    -scipy.special.kv(self.params[1] - 2, q[idxs]) +
                    scipy.special.kv(self.params[1] - 1, q[idxs]) / q[idxs]
                )
                # Had to assume nu > 1 for this to work: CHECK THIS!
                k[idxs & zero_mask] = 2**(self.params[1] - 1) * self.params[1] * scipy.special.gamma(self.params[1] - 1) / (self.params[2]**2)
            else:
                raise NotImplementedError("Derivatives greater than [1, 1] are not supported!")
        k = (self.params[0]**2 * 2**(1 - self.params[1]) / scipy.special.gamma(self.params[1])) * k
        return k

class MaternKernel(ChainRuleKernel):
    r"""Matern covariance kernel. Supports arbitrary derivatives. Treats order as a hyperparameter.
    
    The Matern kernel has the following hyperparameters, always referenced in
    the order listed:
    
    = ===== ====================================
    0 sigma prefactor
    1 nu    order of kernel
    2 l1    length scale for the first dimension
    3 l2    ...and so on for all dimensions
    = ===== ====================================
    
    The kernel is defined as:
    
    .. math::
    
        k_M = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)}
        \left (\sqrt{2\nu \sum_i\left (\frac{\tau_i^2}{l_i^2}\right )}\right )^\nu
        K_\nu\left(\sqrt{2\nu \sum_i\left(\frac{\tau_i^2}{l_i^2}\right)}\right)
    
    Parameters
    ----------
    num_dim : int
        Number of dimensions of the input data. Must be consistent with the `X`
        and `Xstar` values passed to the :py:class:`~gptools.gaussian_process.GaussianProcess`
        you wish to use the covariance kernel with.
    **kwargs
        All keyword parameters are passed to :py:class:`~gptools.kernel.core.ChainRuleKernel`.
    
    Raises
    ------
    ValueError
        If `num_dim` is not a positive integer or the lengths of the input
        vectors are inconsistent.
    GPArgumentError
        If `fixed_params` is passed but `initial_params` is not.
    """
    def __init__(self, num_dim=1, **kwargs):
        param_names = [r'\sigma_f', r'\nu'] + ['l_%d' % (i + 1,) for i in range(0, num_dim)]
        super(MaternKernel, self).__init__(num_dim=num_dim,
                                           num_params=num_dim + 2,
                                           param_names=param_names,
                                           **kwargs)
    
    def _compute_k(self, tau):
        r"""Evaluate the kernel directly at the given values of `tau`.
        
        Parameters
        ----------
        tau : :py:class:`Matrix`, (`M`, `D`)
            `M` inputs with dimension `D`.
        
        Returns
        -------
        k : :py:class:`Array`, (`M`,)
            :math:`k(\tau)` (less the :math:`\sigma^2` prefactor).
        """
        y, r2l2 = self._compute_y(tau, return_r2l2=True)
        k = 2.0**(1.0 - self.nu) / scipy.special.gamma(self.nu) * y**(self.nu / 2.0) * scipy.special.kv(self.nu, scipy.sqrt(y))
        k[r2l2 == 0] = 1.0
        return k
    
    def _compute_y(self, tau, return_r2l2=False):
        r"""Covert tau to :math:`y=2\nu\sum_i(\tau_i^2/l_i^2)`.
        
        Parameters
        ----------
        tau : :py:class:`Matrix`, (`M`, `D`)
            `M` inputs with dimension `D`.
        return_r2l2 : bool, optional
            Set to True to return a tuple of (`y`, `r2l2`). Default is False
            (only return `y`).
        
        Returns
        -------
        y : :py:class:`Array`, (`M`,)
            Inner argument of function.
        r2l2 : :py:class:`Array`, (`M`,)
            Anisotropically scaled distances. Only returned if `return_r2l2` is True.
        """
        r2l2 = self._compute_r2l2(tau)
        y = 2.0 * self.nu * r2l2
        if return_r2l2:
            return (y, r2l2)
        else:
            return y
    
    def _compute_y_wrapper(self, *args):
        r"""Convert tau to :math:`y=\sqrt{2\nu\sum_i(\tau_i^2/l_i^2)}`.
        
        Takes `tau` as an argument list for compatibility with :py:func:`mpmath.diff`.
        
        Parameters
        ----------
        tau[0] : scalar float
            First element of `tau`.
        tau[1] : And so on...
        
        Returns
        -------
        y : scalar float
            Inner part of Matern kernel at the given `tau`.
        """
        return self._compute_y(scipy.atleast_2d(scipy.asarray(args, dtype=float)))
    
    def _compute_dk_dy(self, y, n):
        r"""Evaluate the derivative of the outer form of the Matern kernel.
        
        Uses the general Leibniz rule to compute the n-th derivative of:
        
        .. math::
        
            f(y) = \frac{2^{1-\nu}}{\Gamma(\nu)} y^{\nu/2} K_\nu(y^{1/2})
        
        Parameters
        ----------
        y : :py:class:`Array`, (`M`,)
            `M` inputs to evaluate at.
        n : non-negative scalar int.
            Order of derivative to compute.
        
        Returns
        -------
        dk_dy : :py:class:`Array`, (`M`,)
            Specified derivative at specified locations.
        """
        return 2.0**(1 - self.nu) / (scipy.special.gamma(self.nu)) * yn2Kn2Der(self.nu, y, n=n)
    
    def _compute_dy_dtau(self, tau, b, r2l2):
        r"""Evaluate the derivative of the inner argument of the Matern kernel.
        
        Take the derivative of
        
        .. math::
        
            y = 2 \nu \sum_i(\tau_i^2 / l_i^2)
        
        Parameters
        ----------
        tau : :py:class:`Matrix`, (`M`, `D`)
            `M` inputs with dimension `D`.
        b : :py:class:`Array`, (`P`,)
            Block specifying derivatives to be evaluated.
        r2l2 : :py:class:`Array`, (`M`,)
            Precomputed anisotropically scaled distance.
        
        Returns
        -------
        dy_dtau: :py:class:`Array`, (`M`,)
            Specified derivative at specified locations.
        """
        if len(b) == 0:
            return self._compute_y(tau)
        elif len(b) == 1:
            return 4.0 * self.nu * tau[:, b[0]] / (self.params[2 + b[0]])**2.0
        elif (len(b) == 2) and (b[0] == b[1]):
            return 4.0 * self.nu / (self.params[2 + b[0]])**2.0
        else:
            return scipy.zeros_like(r2l2)
    
    def _compute_dk_dtau_on_partition(self, tau, p):
        """Evaluate the term inside the sum of Faa di Bruno's formula for the given partition.
        
        Overrides the version from :py:class:`gptools.kernel.core.ChainRuleKernel`
        in order to get the correct behavior at the origin.
        
        Parameters
        ----------
        tau : :py:class:`Matrix`, (`M`, `D`)
            `M` inputs with dimension `D`.
        p : list of :py:class:`Array`
            Each element is a block of the partition representing the
            derivative orders to use.
        
        Returns
        -------
        dk_dtau : :py:class:`Array`, (`M`,)
            The specified derivatives over the given partition at the specified
            locations.
        """
        # Find the derivative order:
        n = len(p)
        y, r2l2 = self._compute_y(tau, return_r2l2=True)
        # Keep track of how many times a given variable has a block of length 1:
        n1 = 0
        # Build the dy/dtau factor up iteratively:
        dy_dtau_factor = scipy.ones_like(y)
        for b in p:
            # If the partial derivative is exactly zero there is no sense in
            # continuing the computation:
            if (len(b) > 2) or ((len(b) == 2) and (b[0] != b[1])):
                return scipy.zeros_like(y)
            dy_dtau_factor *= self._compute_dy_dtau(tau, b, r2l2)
            # Count the number of blocks of length 1:
            if len(b) == 1:
                n1 += 1.0
        # Compute d^(|pi|)f/dy^(|pi|) term:
        dk_dy = self._compute_dk_dy(y, n)
        if n1 > 0:
            mask = (y == 0.0)
            tau_pow = 2 * (self.nu - n) + n1
            if tau_pow == 0:
                # In this case the limit does not exist, so it is set to NaN:
                dk_dy[mask] = scipy.nan
            elif tau_pow > 0:
                dk_dy[mask] = 0.0
                
        return dk_dy * dy_dtau_factor
    
    @property
    def nu(self):
        r"""Returns the value of the order :math:`\nu`.
        """
        return self.params[1]


class Matern52Kernel(Kernel):
    r"""Matern 5/2 covariance kernel. Supports only 0th and 1st derivatives
    and is fixed at nu=5/2. Because of these limitations, it is quite a bit
    faster than the more general Matern kernels.

    The Matern52 kernel has the following hyperparameters, always referenced in
    the order listed:

    = ===== ====================================
    0 sigma prefactor
    1 l1    length scale for the first dimension
    2 l2    ...and so on for all dimensions
    = ===== ====================================

    The kernel is defined as:

    .. math::

        k_M(x, x') = \sigma^2 \left(1 + \sqrt{5r^2} + \frac{5}{3}r^2\right) \exp(-\sqrt{5r^2}) \\
        r^2 = \sum_{d=1}^D \frac{(x_d - x'_d)^2}{l_d^2}

    Parameters
    ----------
    num_dim : int
        Number of dimensions of the input data. Must be consistent with the `X`
        and `Xstar` values passed to the :py:class:`~gptools.gaussian_process.GaussianProcess`
        you wish to use the covariance kernel with.
    **kwargs
        All keyword parameters are passed to :py:class:`~gptools.kernel.core.Kernel`.

    Raises
    ------
    ValueError
        If `num_dim` is not a positive integer or the lengths of the input
        vectors are inconsistent.
    GPArgumentError
        If `fixed_params` is passed but `initial_params` is not.
    """
    def __init__(self, num_dim=1, **kwargs):
        param_names = [r'\sigma_f'] + ['l_%d' % (i + 1,) for i in range(0, num_dim)]
        super(Matern52Kernel, self).__init__(num_dim=num_dim,
                                             num_params=num_dim + 1,
                                             param_names=param_names,
                                             **kwargs)
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
        symmetric : bool
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
        if scipy.any(scipy.sum(ni, axis=1) > 1) or scipy.any(scipy.sum(nj, axis=1) > 1):
            raise ValueError("Matern52Kernel only supports 0th and 1st order derivatives")

        Xi = scipy.asarray(Xi, dtype=scipy.float64)
        Xj = scipy.asarray(Xj, dtype=scipy.float64)
        ni = scipy.array(ni, dtype=scipy.int32)
        nj = scipy.array(nj, dtype=scipy.int32)
        var = scipy.square(self.params[-self.num_dim:])

        value = _matern52(Xi, Xj, ni, nj, var)
        return self.params[0]**2 * value
