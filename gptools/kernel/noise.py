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

"""Provides classes for implementing uncorrelated noise.
"""

from __future__ import division

from .core import Kernel

import scipy

class DiagonalNoiseKernel(Kernel):
    """Kernel that has constant, independent noise (i.e., a diagonal kernel).
    
    Parameters
    ----------
    num_dim : positive int
        Number of dimensions of the input data.
    initial_noise : float, optional
        Initial value for the noise standard deviation. Default value is None
        (noise gets set to 1).
    fixed_noise : bool, optional
        Whether or not the noise is taken to be fixed when optimizing the log
        likelihood. Default is False (noise is not fixed).
    noise_bound : 2-tuple, optional
        The bounds for the noise when optimizing the log likelihood with
        :py:func:`scipy.optimize.minimize`. Must be of the form
        (`lower`, `upper`). Set a given entry to None to put no bound on
        that side. Default is None, which gets set to (0, None).
    n : non-negative int or tuple of non-negative ints with length equal to `num_dim`, optional
        Indicates which derivative this noise is with respect to. Default is 0
        (noise applies to value).
    hyperprior : callable, optional
        Function that returns the prior log-density for a possible value of
        noise when called. Must also have an attribute called :py:attr:`bounds`
        that is the bounds on the noise and a method called
        :py:meth:`random_draw` that yields a random draw. Default behavior is
        to assign a uniform prior.
    """
    def __init__(self, num_dim=1, initial_noise=None, fixed_noise=False, noise_bound=None, n=0, hyperprior=None):
        try:
            iter(n)
        except TypeError:
            self.n = n * scipy.ones(num_dim, dtype=int)
        else:
            if len(n) != num_dim:
                raise ValueError("Length of n must be equal to num_dim!")
            self.n = scipy.asarray(n, dtype=int)
        if initial_noise is not None:
            initial_noise = [initial_noise]
        if noise_bound is not None:
            noise_bound = [noise_bound]
        super(DiagonalNoiseKernel, self).__init__(num_dim=num_dim,
                                                  num_params=1,
                                                  initial_params=initial_noise,
                                                  fixed_params=[fixed_noise],
                                                  param_bounds=noise_bound,
                                                  hyperprior=hyperprior,
                                                  param_names=[r'\sigma_n'])
    
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
            with respect to. Since this kernel only has one hyperparameter, 0
            is the only valid value. If None, no derivatives are taken. Default
            is None (no hyperparameter derivatives).
        symmetric : bool, optional
            Whether or not the input `Xi`, `Xj` are from a symmetric matrix.
            Default is False.
        
        Returns
        -------
        Kij : :py:class:`Array`, (`M`,)
            Covariances for each of the `M` `Xi`, `Xj` pairs.
        """
        if symmetric:
            val = self.params[0]**2 * scipy.asarray(((Xi == Xj) & (ni == self.n) & (nj == self.n)).all(axis=1), dtype=float).flatten()
            if hyper_deriv is None:
                return val
            else:
                return 2.0 * val / self.params[hyper_deriv]
        else:
            return scipy.zeros(Xi.shape[0])

class ZeroKernel(DiagonalNoiseKernel):
    """Kernel that always evaluates to zero, used as the default noise kernel.
    
    Parameters
    ----------
    num_dim : positive int
        The number of dimensions of the inputs.
    """
    def __init__(self, num_dim=1):
        super(ZeroKernel, self).__init__(num_dim=num_dim, initial_noise=0.0, fixed_noise=True)

    def __call__(self, Xi, Xj, ni, nj, hyper_deriv=None, symmetric=False):
        """Return zeros the same length as the input Xi.
        
        Ignores all other arguments.
        
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
            with respect to. Since this kernel only has one hyperparameter, 0
            is the only valid value. If None, no derivatives are taken. Default
            is None (no hyperparameter derivatives).
        symmetric : bool, optional
            Whether or not the input `Xi`, `Xj` are from a symmetric matrix.
            Default is False.
        
        Returns
        -------
        Kij : :py:class:`Array`, (`M`,)
            Covariances for each of the `M` `Xi`, `Xj` pairs.
        """
        return scipy.zeros(Xi.shape[0], dtype=float)
