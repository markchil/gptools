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

"""Provides the :py:class:`RationalQuadraticKernel` class which implements the anisotropic rational quadratic (RQ) kernel.
"""

from __future__ import division

from .core import ChainRuleKernel

import scipy
import scipy.special
import scipy.misc

class RationalQuadraticKernel(ChainRuleKernel):
    r"""Rational quadratic (RQ) covariance kernel. Supports arbitrary derivatives.
    
    The RQ kernel has the following hyperparameters, always referenced
    in the order listed:
    
    = ===== =====================================
    0 sigma prefactor.
    1 alpha order of kernel.
    2 l1    length scale for the first dimension.
    3 l2    ...and so on for all dimensions.
    = ===== =====================================
    
    The kernel is defined as:
    
    .. math::
    
        k_{RQ} = \sigma^2 \left(1 + \frac{1}{2\alpha} \sum_i\frac{\tau_i^2}{l_i^2}\right)^{-\alpha}

    Parameters
    ----------
    num_dim : int
        Number of dimensions of the input data. Must be consistent
        with the `X` and `Xstar` values passed to the
        :py:class:`~gptools.gaussian_process.GaussianProcess` you
        wish to use the covariance kernel with.
    **kwargs
        All keyword parameters are passed to :py:class:`~gptools.kernel.core.ChainRuleKernel`.

    Raises
    ------
    ValueError
        If `num_dim` is not a positive integer or the lengths of
        the input vectors are inconsistent.
    GPArgumentError
        If `fixed_params` is passed but `initial_params` is not.
    """
    def __init__(self, num_dim=1, **kwargs):
        param_names = [r'\sigma_f', r'\alpha'] + ['l_%d' % (i + 1,) for i in range(0, num_dim)]
        super(RationalQuadraticKernel, self).__init__(num_dim=num_dim,
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
        y = self._compute_y(tau)
        return y**(-self.params[1])
    
    def _compute_y(self, tau, return_r2l2=False):
        r"""Covert tau to :math:`y = 1 + \frac{1}{2\alpha} \sum_i \frac{\tau_i^2}{l_i^2}`.
        
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
            Anisotropically scaled distances. Only returned if `return_r2l2`
            is True.
        """
        r2l2 = self._compute_r2l2(tau)
        y = 1.0 + 1.0 / (2.0 * self.params[1]) * r2l2
        if return_r2l2:
            return (y, r2l2)
        else:
            return y
    
    def _compute_dk_dy(self, y, n):
        """Evaluate the derivative of the outer form of the RQ kernel.
        
        Parameters
        ----------
        y : :py:class:`Array`, (`M`,)
            `M` inputs to evaluate at.
        n : non-negative scalar int
            Order of derivative to compute.
        
        Returns
        -------
        dk_dy : :py:class:`Array`, (`M`,)
            Specified derivative at specified locations.
        """
        # Need conditional statement because scipy's impelementation of the
        # Pochhammer symbol is wrong for negative integer arguments:
        # Uses the definition from
        # http://functions.wolfram.com/GammaBetaErf/Pochhammer/02/
        a = 1.0 - self.params[1] - n
        if a < 0.0 and a % 1 == 0 and n <= -a:
            p = (-1.0)**n * scipy.misc.factorial(-a) / scipy.misc.factorial(-a - n)
        else:
            p = scipy.special.poch(a, n)
        return p * y**(-self.params[1] - n)
    
    def _compute_dy_dtau(self, tau, b, r2l2):
        r"""Evaluate the derivative of the inner argument of the Matern kernel.
        
        Uses Faa di Bruno's formula to take the derivative of
        
        .. math::
        
            y = 1 + \frac{1}{2\alpha}\sum_i(\tau_i^2/l_i^2)}`.
        
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
        dy_dtau : :py:class:`Array`, (`M`,)
            Specified derivative at specified locations.
        """
        if len(b) == 0:
            return self._compute_y(tau)
        elif len(b) == 1:
            return 1.0 / self.params[1] * tau[:, b[0]] / (self.params[2 + b[0]])**2.0
        elif len(b) == 2 and b[0] == b[1]:
            return 1.0 / (self.params[1] * (self.params[2 + b[0]])**2.0)
        else:
            return scipy.zeros_like(r2l2)
