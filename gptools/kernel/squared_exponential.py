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

"""Provides the :py:class:`SquaredExponentialKernel` class that implements the anisotropic SE kernel.
"""

from __future__ import division

from .core import Kernel

import scipy
import scipy.special

class SquaredExponentialKernel(Kernel):
    r"""Squared exponential covariance kernel. Supports arbitrary derivatives.
    
    The squared exponential has the following hyperparameters, always
    referenced in the order listed:
    
    = ===== ====================================
    0 sigma prefactor on the SE
    1 l1    length scale for the first dimension
    2 l2    ...and so on for all dimensions
    = ===== ====================================
    
    The kernel is defined as:
    
    .. math::
    
        k_{SE} = \sigma^2 \exp\left(-\frac{1}{2}\sum_i\frac{\tau_i^2}{l_i^2}\right)
    
    Parameters
    ----------
    num_dim : int
        Number of dimensions of the input data. Must be consistent
        with the `X` and `Xstar` values passed to the
        :py:class:`~gptools.gaussian_process.GaussianProcess` you
        wish to use the covariance kernel with.
    **kwargs
        All keyword parameters are passed to :py:class:`~gptools.kernel.core.Kernel`.
    
    Raises
    ------
    ValueError
        If `num_dim` is not a positive integer or the lengths of
        the input vectors are inconsistent.
        
    GPArgumentError
        If `fixed_params` is passed but `initial_params` is not.
    """
    def __init__(self, num_dim=1, **kwargs):
        param_names = [r'\sigma_f'] + ['l_%d' % (i + 1,) for i in range(0, num_dim)]
        super(SquaredExponentialKernel, self).__init__(num_dim=num_dim,
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
            with respect to. If None, no derivatives are taken. Default is None
            (no hyperparameter derivatives). Hyperparameter derivatives are not
            support for `n` > 0 at this time.
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
            If hyper_deriv is not None and `n` > 0.
        """
        only_first_order = ((scipy.asarray(ni, dtype=int) == 0).all() and
                            (scipy.asarray(nj, dtype=int) == 0).all())
        if hyper_deriv is not None and not only_first_order:
            raise NotImplementedError("Hyperparameter derivatives with n > 0 "
                                      "have not been implemented!")
        tau = scipy.asarray(Xi - Xj, dtype=float)
        r2l2, l_mat = self._compute_r2l2(tau, return_l=True)
        k = self.params[0]**2 * scipy.exp(-r2l2 / 2.0)
        # Account for derivatives:
        # Get total number of differentiations:
        n_tot_j = scipy.asarray(scipy.sum(nj, axis=1), dtype=int).flatten()
        n_combined = scipy.asarray(ni + nj, dtype=int)
        # Compute factor from the dtau_d/dx_d_j terms in the chain rule:
        j_chain_factors = (-1.0)**(n_tot_j)
        # Compute Hermite polynomial factor:
        hermite_factors = scipy.prod((-1.0 / (scipy.sqrt(2.0) * l_mat))**(n_combined) *
                                     scipy.special.eval_hermite(n_combined, tau / (scipy.sqrt(2.0) * l_mat)),
                                     axis=1)
        
        k = j_chain_factors * hermite_factors * k
        # Take care of hyperparameter derivatives:
        if hyper_deriv is None:
            return k
        elif hyper_deriv == 0:
            return 2 * k / self.params[0]
        else:
            return (tau[:, hyper_deriv - 1])**2 / (self.params[hyper_deriv - 1])**3 * k
