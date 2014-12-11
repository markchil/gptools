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

"""Core kernel classes: contains the base :py:class:`Kernel` class and helper subclasses.
"""

from __future__ import division

from ..utils import unique_rows, generate_set_partitions, UniformJointPrior, \
                    ProductJointPrior, IndependentJointPrior, powerset, MaskedBounds
from ..error_handling import GPArgumentError

import scipy
import scipy.special
import scipy.stats
try:
    import mpmath
except ImportError:
    import warnings
    warnings.warn("Could not import mpmath. ArbitraryKernel class will not work.",
                  ImportWarning)
import inspect
import multiprocessing

class Kernel(object):
    """Covariance kernel base class. Not meant to be explicitly instantiated!
    
    Initialize the kernel with the given number of input dimensions.
    
    Parameters
    ----------
    num_dim : positive int
        Number of dimensions of the input data. Must be consistent with the `X`
        and `Xstar` values passed to the :py:class:`~gptools.gaussian_process.GaussianProcess`
        you wish to use the covariance kernel with. Default is 1.
    num_params : Non-negative int
        Number of parameters in the model.
    initial_params : :py:class:`Array` or other Array-like, (`num_params`,), optional
        Initial values to set for the hyperparameters. Default is None, in
        which case 1 is used for the initial values.
    fixed_params : :py:class:`Array` or other Array-like of bool, (`num_params`,), optional
        Sets which hyperparameters are considered fixed when optimizing the log
        likelihood. A True entry corresponds to that element being
        fixed (where the element ordering is as defined in the class).
        Default value is None (no hyperparameters are fixed).
    param_bounds : list of 2-tuples (`num_params`,), optional
        List of bounds for each of the hyperparameters. Each 2-tuple is of the
        form (lower`, `upper`). If there is no bound in a given direction, it
        works best to set it to something big like 1e16. Default is (0.0, 1e16)
        for each hyperparameter. Note that this is overridden by the `hyperprior`
        keyword, if present.
    param_names : list of str (`num_params`,), optional
        List of labels for the hyperparameters. Default is all empty strings.
    enforce_bounds : bool, optional
        If True, an attempt to set a hyperparameter outside of its bounds will
        result in the hyperparameter being set right at its bound. If False,
        bounds are not enforced inside the kernel. Default is False (do not
        enforce bounds).
    hyperprior : :py:class:`JointPrior` instance or list, optional
        Joint prior distribution for all hyperparameters. Can either be given
        as a :py:class:`JointPrior` instance or a list of `num_params`
        callables or py:class:`rv_frozen` instances from :py:mod:`scipy.stats`,
        in which case a :py:class:`IndependentJointPrior` is constructed with
        these as the independent priors on each hyperparameter. Default is a
        uniform PDF on all hyperparameters.
    
    Attributes
    ----------
    num_params : int
        Number of parameters
    num_dim : int
        Number of dimensions
    params : :py:class:`Array` of float, (`num_params`,)
        Array of parameters.
    fixed_params : :py:class:`Array` of bool, (`num_params`,)
        Array of booleans indicated which parameters in params are fixed.
    param_names : list of str, (`num_params`,)
        List of the labels for the hyperparameters.
    hyperprior : :py:class:`JointPrior` instance
        Joint prior distribution for the hyperparameters.
    
    Raises
    ------
    ValueError
        If `num_dim` is not a positive integer or the lengths of the input
        vectors are inconsistent.
        
    GPArgumentError
        if `fixed_params` is passed but `initial_params` is not.
    """
    def __init__(self, num_dim=1, num_params=0, initial_params=None,
                 fixed_params=None, param_bounds=None, param_names=None,
                 enforce_bounds=False, hyperprior=None):
        if num_params < 0 or not isinstance(num_params, (int, long)):
            raise ValueError("num_params must be an integer >= 0!")
        self.num_params = num_params
        if param_names is None:
            param_names = [''] * self.num_params
        elif len(param_names) != self.num_params:
            raise ValueError("param_names must be a list of length num_params!")
        self.param_names = param_names
        
        if num_dim < 1 or not isinstance(num_dim, (int, long)):
            raise ValueError("num_dim must be an integer > 0!")
        self.num_dim = num_dim
        
        self.enforce_bounds = enforce_bounds
        
        # Handle default case for initial parameter values -- set them all to 1.
        if initial_params is None:
            # Only accept fixed_params if initial_params is given:
            if fixed_params is not None:
                raise GPArgumentError("Must pass explicit parameter values "
                                      "if fixing parameters!")
            initial_params = scipy.ones(num_params, dtype=float)
            fixed_params = scipy.zeros(num_params, dtype=float)
        else:
            if len(initial_params) != num_params:
                raise ValueError("Length of initial_params must be equal to num_params!")
            # Handle default case of fixed_params: no fixed parameters.
            if fixed_params is None:
                fixed_params = scipy.zeros(num_params, dtype=float)
            else:
                if len(fixed_params) != num_params:
                    raise ValueError("Length of fixed_params must be equal to num_params!")
        
        # Handle default case for parameter bounds -- set them all to (0, 1e16):
        if param_bounds is None:
            param_bounds = num_params * [(0.0, 1e16)]
        else:
            if len(param_bounds) != num_params:
                raise ValueError("Length of param_bounds must be equal to num_params!")
        
        # Handle default case for hyperpriors -- set them all to be uniform:
        if hyperprior is None:
            hyperprior = UniformJointPrior(param_bounds)
        else:
            try:
                iter(hyperprior)
                if len(hyperprior) != num_params:
                    raise ValueError("If hyperprior is a list its length must "
                                     "be equal to num_params!")
                hyperprior = IndependentJointPrior(hyperprior)
            except TypeError:
                pass
        
        self.params = scipy.asarray(initial_params, dtype=float)
        self.fixed_params = scipy.asarray(fixed_params, dtype=bool)
        self.hyperprior = hyperprior
    
    @property
    def param_bounds(self):
        return self.hyperprior.bounds
    
    @param_bounds.setter
    def param_bounds(self, value):
        self.hyperprior.bounds = value
    
    def __call__(self, Xi, Xj, ni, nj, hyper_deriv=None, symmetric=False):
        """Evaluate the covariance between points `Xi` and `Xj` with derivative order `ni`, `nj`.
        
        Note that this method only returns the covariance -- the hyperpriors
        and potentials stored in this kernel must be applied separately.
        
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
            (no hyperparameter derivatives).
        symmetric : bool, optional
            Whether or not the input `Xi`, `Xj` are from a symmetric matrix.
            Default is False.
        
        Returns
        -------
        Kij : :py:class:`Array`, (`M`,)
            Covariances for each of the `M` `Xi`, `Xj` pairs.
        
        Notes
        -----
        THIS IS ONLY A METHOD STUB TO DEFINE THE NEEDED CALLING FINGERPRINT!
        """
        raise NotImplementedError("This is an abstract method -- please use "
                                  "one of the implementing subclasses!")
    
    def set_hyperparams(self, new_params):
        """Sets the free hyperparameters to the new parameter values in new_params.
        
        Parameters
        ----------
        new_params : :py:class:`Array` or other Array-like, (len(:py:attr:`self.params`),)
            New parameter values, ordered as dictated by the docstring for the
            class.
        """
        new_params = scipy.asarray(new_params, dtype=float)
        
        if len(new_params) == len(self.free_params):
            if self.enforce_bounds:
                for idx, new_param, bound in zip(range(0, len(new_params)), new_params, self.free_param_bounds):
                    if bound[0] is not None and new_param < bound[0]:
                        new_params[idx] = bound[0]
                    elif bound[1] is not None and new_param > bound[1]:
                        new_params[idx] = bound[1]
            self.params[~self.fixed_params] = new_params
        else:
            raise ValueError("Length of new_params must be %s!" % (len(self.free_params),))
    
    @property
    def num_free_params(self):
        """Returns the number of free parameters.
        """
        return sum(~self.fixed_params)
    
    @property
    def free_param_idxs(self):
        """Returns the indices of the free parameters in the main arrays of parameters, etc.
        """
        return scipy.arange(0, self.num_params)[~self.fixed_params]
    
    @property
    def free_params(self):
        """Returns the values of the free hyperparameters.
        
        Returns
        -------
        free_params : :py:class:`Array`
            Array of the free parameters, in order.
        """
        return MaskedBounds(self.params, self.free_param_idxs)
    
    @free_params.setter
    def free_params(self, value):
        self.params[self.free_param_idxs] = scipy.asarray(value, dtype=float)
    
    @property
    def free_param_bounds(self):
        """Returns the bounds of the free hyperparameters.
        
        Returns
        -------
        free_param_bounds : :py:class:`Array`
            Array of the bounds of the free parameters, in order.
        """
        return MaskedBounds(self.hyperprior.bounds, self.free_param_idxs)
    
    @free_param_bounds.setter
    def free_param_bounds(self, value):
        # Need to use a loop since self.hyperprior.bounds is NOT guaranteed to support fancy indexing.
        for i, v in zip(self.free_param_idxs, value):
            self.hyperprior.bounds[i] = v
    
    @property
    def free_param_names(self):
        """Returns the names of the free hyperparameters.
        
        Returns
        -------
        free_param_names : :py:class:`Array`
            Array of the names of the free parameters, in order.
        """
        return MaskedBounds(self.param_names, self.free_param_idxs)
    
    @free_param_names.setter
    def free_param_names(self, value):
        # Cast to array in case it hasn't been done already:
        self.param_names = scipy.asarray(self.param_names, dtype=str)
        self.param_names[~self.fixed_params] = value
    
    def __add__(self, other):
        """Add two Kernels together.
        
        Parameters
        ----------
        other : :py:class:`Kernel`
            Kernel to be added to this one.
        
        Returns
        -------
        sum : :py:class:`SumKernel`
            Instance representing the sum of the two kernels.
        """
        return SumKernel(self, other)
    
    def __mul__(self, other):
        """Multiply two Kernels together.
        
        Parameters
        ----------
        other : :py:class:`Kernel`
            Kernel to be multiplied by this one.
        
        Returns
        -------
        prod : :py:class:`ProductKernel`
            Instance representing the product of the two kernels.
        """
        return ProductKernel(self, other)
    
    def _compute_r2l2(self, tau, return_l=False):
        r"""Compute the anisotropic :math:`r^2/l^2` term for the given `tau`.
        
        Here, :math:`\tau=X_i-X_j` is the difference vector. Computes
        .. math::
            \frac{r^2}{l^2} = \sum_i\frac{\tau_i^2}{l_{i}^{2}}
        Assumes that the length parameters are the last `num_dim` elements of
        :py:attr:`self.params`.
        
        Where `l` and `tau` are both zero, that term is set to zero.
        
        Parameters
        ----------
        tau : :py:class:`Array`, (`M`, `D`)
            `M` inputs with dimension `D`.
        return_l : bool, optional
            Set to True to return a tuple of (`tau`, `l_mat`), where `l_mat`
            is the matrix of length scales to match the shape of `tau`. Default
            is False (only return `tau`).
        
        Returns
        -------
        r2l2 : :py:class:`Array`, (`M`,)
            Anisotropically scaled distances squared.
        l_mat : :py:class:`Array`, (`M`, `D`)
            The (`D`,) array of length scales repeated for each of the `M`
            inputs. Only returned if `return_l` is True.
        """
        l_mat = scipy.tile(self.params[-self.num_dim:], (tau.shape[0], 1))
        tau_over_l = tau / l_mat
        tau_over_l[(tau == 0) & (l_mat == 0)] = 0.0
        r2l2 = scipy.sum((tau_over_l)**2, axis=1)
        if return_l:
            return (r2l2, l_mat)
        else:
            return r2l2

class BinaryKernel(Kernel):
    """Abstract class for binary operations on kernels (addition, multiplication, etc.).
    
    Parameters
    ----------
    k1, k2 : :py:class:`Kernel` instances to be combined
    
    Notes
    -----
    `k1` and `k2` must have the same number of dimensions.
    """
    def __init__(self, k1, k2):
        if not isinstance(k1, Kernel) or not isinstance(k2, Kernel):
            raise TypeError("Both arguments to BinaryKernel must be instances of "
                            "type Kernel!")
        if k1.num_dim != k2.num_dim:
            raise ValueError("Only kernels having the same number of dimensions "
                             "can be summed!")
        self.k1 = k1
        self.k2 = k2
        
        self._enforce_bounds = k1.enforce_bounds or k2.enforce_bounds
        
        super(BinaryKernel, self).__init__(num_dim=k1.num_dim,
                                           num_params=k1.num_params + k2.num_params,
                                           initial_params=scipy.concatenate((k1.params, k2.params)),
                                           fixed_params=scipy.concatenate((k1.fixed_params, k2.fixed_params)),
                                           param_names=list(k1.param_names) + list(k2.param_names),
                                           hyperprior=k1.hyperprior * k2.hyperprior,
                                           enforce_bounds=self._enforce_bounds)
    
    @property
    def enforce_bounds(self):
        """Boolean indicating whether or not the kernel will explicitly enforce the bounds defined by the hyperprior.
        """
        return self._enforce_bounds
    
    @enforce_bounds.setter
    def enforce_bounds(self, v):
        """Set `enforce_bounds` for both of the kernels to a new value.
        """
        self._enforce_bounds = v
        self.k1.enforce_bounds = v
        self.k2.enforce_bounds = v
    
    @property
    def fixed_params(self):
        return scipy.concatenate((self.k1.fixed_params, self.k2.fixed_params))
    
    @fixed_params.setter
    def fixed_params(self, v):
        self.k1.fixed_params = v[:self.k1.num_params]
        self.k2.fixed_params = v[self.k1.num_params:]
    
    @property
    def free_param_bounds(self):
        """Returns the bounds of the free hyperparameters.
        
        Returns
        -------
        free_param_bounds : :py:class:`Array`
            Array of the bounds of the free parameters, in order.
        """
        return scipy.concatenate((self.k1.free_param_bounds, self.k2.free_param_bounds))
    
    @property
    def free_param_names(self):
        """Returns the names of the free hyperparameters.
        
        Returns
        -------
        free_param_names : :py:class:`Array`
            Array of the names of the free parameters, in order.
        """
        return scipy.concatenate((self.k1.free_param_names, self.k2.free_param_names))
    
    @property
    def params(self):
        return scipy.concatenate((self.k1.params, self.k2.params))
    
    @params.setter
    def params(self, v):
        self.k1.params = v[:self.k1.num_params]
        self.k2.params = v[self.k1.num_params:]
    
    def set_hyperparams(self, new_params):
        """Set the (free) hyperparameters.
        
        Parameters
        ----------
        new_params : :py:class:`Array` or other Array-like
            New values of the free parameters.
        
        Raises
        ------
        ValueError
            If the length of `new_params` is not consistent with :py:attr:`self.params`.
        """
        new_params = scipy.asarray(new_params, dtype=float)
        
        if len(new_params) == len(self.free_params):
            num_free_k1 = sum(~self.k1.fixed_params)
            self.k1.set_hyperparams(new_params[:num_free_k1])
            self.k2.set_hyperparams(new_params[num_free_k1:])
        else:
            raise ValueError("Length of new_params must be %s!" % (len(self.free_params),))

class SumKernel(BinaryKernel):
    """The sum of two kernels.
    """
    def __call__(self, *args, **kwargs):
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
            If the `hyper_deriv` keyword is given and is not None.
        """
        if 'hyper_deriv' in kwargs and kwargs['hyper_deriv'] is not None:
            raise NotImplementedError("Keyword hyper_deriv is not presently "
                                      "supported for SumKernel!")
        return self.k1(*args, **kwargs) + self.k2(*args, **kwargs)

class ProductKernel(BinaryKernel):
    """The product of two kernels.
    """
    def __call__(self, Xi, Xj, ni, nj, **kwargs):
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
            If the `hyper_deriv` keyword is given and is not None.
        """
        # Need to process ni, nj to handle the product rule properly.
        nij = scipy.hstack((ni, nj))
        nij_unique = unique_rows(nij)
        
        result = scipy.zeros(Xi.shape[0])
        
        for row in nij_unique:
            # deriv_pattern is the pattern of partial derivatives, where the
            # indicies for derivatives with respect to the elements of Xj have
            # been offset by self.num_dim. For instance, if ni = [1, 2] and
            # nj = [3, 4], deriv_pattern will be [0, 1, 1, 2, 2, 2, 3, 3, 3, 3].
            deriv_pattern = []
            for idx in xrange(0, len(row)):
                deriv_pattern.extend(row[idx] * [idx])
            
            idxs = (nij == row).all(axis=1)
            
            S = powerset(deriv_pattern)
            
            # little "s" is a member of the power set of S:
            for s in S:
                # nij_1 is the combined array of derivative orders for function 1:
                nij_1 = scipy.zeros((idxs.sum(), 2 * self.num_dim))
                # sC is the complement of s with respect to S:
                sC = list(deriv_pattern)
                for i in s:
                    nij_1[:, i] += 1
                    sC.remove(i)
                # nij_2 is the combined array of derivative orders for function 2:
                nij_2 = scipy.zeros((idxs.sum(), 2 * self.num_dim))
                for i in sC:
                    nij_2[:, i] += 1
                result[idxs] += (
                    self.k1(Xi[idxs, :], Xj[idxs, :], nij_1[:, :self.num_dim], nij_1[:, self.num_dim:], **kwargs) *
                    self.k2(Xi[idxs, :], Xj[idxs, :], nij_2[:, :self.num_dim], nij_2[:, self.num_dim:], **kwargs)
                )
        return result

class ChainRuleKernel(Kernel):
    """Abstract class for the common methods in creating kernels that require application of Faa di Bruno's formula.
    """
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

        tau = scipy.asarray(Xi - Xj, dtype=float)

        # Account for derivatives:
        # Get total number of differentiations:
        n_tot_j = scipy.asarray(scipy.sum(nj, axis=1), dtype=int).flatten()
        n_combined = scipy.asarray(ni + nj, dtype=int)
        n_combined_unique = unique_rows(n_combined)

        # Evaluate the kernel:
        k = scipy.zeros(Xi.shape[0], dtype=float)
        # First compute dk/dtau
        for n_combined_state in n_combined_unique:
            idxs = (n_combined == n_combined_state).all(axis=1)
            k[idxs] = self._compute_dk_dtau(tau[idxs], n_combined_state)
        
        # Compute factor from the dtau_d/dx_d_j terms in the chain rule:
        j_chain_factors = (-1.0)**(n_tot_j)
        
        # Multiply by the chain rule factor to get dk/dXi or dk/dXj:
        k = (self.params[0])**2.0 * j_chain_factors * k
        return k
    
    def _compute_dk_dtau(self, tau, n):
        r"""Evaluate :math:`dk/d\tau` at the specified locations with the specified derivatives.

        Parameters
        ----------
        tau : :py:class:`Matrix`, (`M`, `D`)
            `M` inputs with dimension `D`.
        n : :py:class:`Array`, (`D`,)
            Degree of derivative with respect to each dimension.

        Returns
        -------
            dk_dtau : :py:class:`Array`, (`M`,)
                Specified derivative at specified locations.
        """
        # Construct the derivative pattern:
        # For each dimension, this will contain the index of the dimension
        # repeated a number of times equal to the order of derivative with
        # respect to that dimension.
        # Example: For d^3 k(x, y, z) / dx^2 dy, n would be [2, 1, 0] and
        # deriv_pattern should be [0, 0, 1]. For k(x, y, z) deriv_pattern is [].
        deriv_pattern = []
        for idx in xrange(0, len(n)):
            deriv_pattern.extend(n[idx] * [idx])
        deriv_pattern = scipy.asarray(deriv_pattern, dtype=int)
        # Handle non-derivative case separately for efficiency:
        if len(deriv_pattern) == 0:
            return self._compute_k(tau)
        else:
            # Compute all partitions of the deriv_pattern:
            deriv_partitions = generate_set_partitions(deriv_pattern)
            # Compute the requested derivative using the multivariate Faa di Bruno's equation:
            dk_dtau = scipy.zeros(tau.shape[0])
            # Loop over the partitions:
            for partition in deriv_partitions:
                dk_dtau += self._compute_dk_dtau_on_partition(tau, partition)
            return dk_dtau
    
    def _compute_dk_dtau_on_partition(self, tau, p):
        """Evaluate the term inside the sum of Faa di Bruno's formula for the given partition.

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
        y, r2l2 = self._compute_y(tau, return_r2l2=True)
        # Compute the d^(|pi|)f/dy term:
        dk_dtau = self._compute_dk_dy(y, len(p))
        # Multiply in each of the block terms:
        for b in p:
            dk_dtau *= self._compute_dy_dtau(tau, b, r2l2)
        return dk_dtau

class ArbitraryKernel(Kernel):
    """Covariance kernel from an arbitrary covariance function.
    
    Computes derivatives using :py:func:`mpmath.diff` and is hence in general
    much slower than a hard-coded implementation of a given kernel.
    
    Parameters
    ----------
    num_dim : positive int
        Number of dimensions of the input data. Must be consistent with the `X`
        and `Xstar` values passed to the
        :py:class:`~gptools.gaussian_process.GaussianProcess` you wish to use
        the covariance kernel with.
    cov_func : callable, takes >= 2 args
        Covariance function. Must take arrays of `Xi` and `Xj` as the
        first two arguments. The subsequent (scalar) arguments are the
        hyperparameters. The number of parameters is found by inspection of
        `cov_func` itself, or with the num_params keyword.
    num_proc : int or None, optional
        Number of procs to use in evaluating covariance derivatives. 0 means
        to do it in serial, None means to use all available cores. Default is
        0 (serial evaluation).
    num_params : int or None, optional
        Number of hyperparameters. If None, inspection will be used to infer
        the number of hyperparameters (but will fail if you used clever business
        with *args, etc.). Default is None (use inspection to find argument
        count).
    **kwargs
        All other keyword parameters are passed to :py:class:`~gptools.kernel.core.Kernel`.
    
    Attributes
    ----------
    cov_func : callable
        The covariance function
    num_proc : non-negative int
        Number of processors to use in evaluating covariance derivatives. 0 means serial.
    """
    def __init__(self, cov_func, num_dim=1, num_proc=0, num_params=None, **kwargs):
        if num_proc is None:
            num_proc = multiprocessing.cpu_count()
        self.num_proc = num_proc
        if num_params is None:
            try:
                argspec = inspect.getargspec(cov_func)[0]
                num_params = len(argspec) - 2
                param_names = argspec[2:]
            except TypeError:
                # Need to remove self from the arg list for bound method:
                argspec = inspect.getargspec(cov_func.__call__)[0]
                num_params = len(argspec) - 3
                param_names = argspec[3:]
        self.cov_func = cov_func
        super(ArbitraryKernel, self).__init__(num_dim=num_dim,
                                              num_params=num_params,
                                              param_names=kwargs.pop('param_names', None),
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
        n_cat = scipy.asarray(scipy.concatenate((ni, nj), axis=1), dtype=int)
        X_cat = scipy.asarray(scipy.concatenate((Xi, Xj), axis=1), dtype=float)
        n_cat_unique = unique_rows(n_cat)
        k = scipy.zeros(Xi.shape[0], dtype=float)
        # Loop over unique derivative patterns:
        if self.num_proc > 1:
            pool = multiprocessing.Pool(processes=self.num_proc)
        for n_cat_state in n_cat_unique:
            idxs = scipy.where(scipy.asarray((n_cat == n_cat_state).all(axis=1)).squeeze())[0]
            if (n_cat_state == 0).all():
                k[idxs] = self.cov_func(Xi[idxs, :], Xj[idxs, :], *self.params)
            else:
                if self.num_proc > 1 and len(idxs) > 1:
                    k[idxs] = scipy.asarray(
                        pool.map(_ArbitraryKernelEval(self, n_cat_state), X_cat[idxs, :]),
                        dtype=float
                    )
                else:
                    for idx in idxs:
                        k[idx] = mpmath.chop(mpmath.diff(self._mask_cov_func,
                                                         X_cat[idx, :],
                                                         n=n_cat_state,
                                                         singular=True))
        
        if self.num_proc > 0:
            pool.close()
        return k
    
    def _mask_cov_func(self, *args):
        """Masks the covariance function into a form usable by :py:func:`mpmath.diff`.
        
        Parameters
        ----------
        *args : `num_dim` * 2 floats
            The individual elements of Xi and Xj to be passed to :py:attr:`cov_func`.
        """
        # Have to do it in two cases to get the 1d unwrapped properly:
        if self.num_dim == 1:
            return self.cov_func(args[0], args[1], *self.params)
        else:
            return self.cov_func(args[:self.num_dim], args[self.num_dim:], *self.params)

class _ArbitraryKernelEval(object):
    """Helper class to support parallel evaluation of the :py:class:ArbitraryKernel:.
    
    Parameters
    ----------
    obj : :py:class:`Kernel` instance
        Instance to wrap to allow parallel computation of.
    n_cat_state : Array-like, (2,)
        Derivative orders to take with respect to `Xi` and `Xj`.
    """
    # TODO: Generalize this for higher dimensions, since ArbitraryKernel is
    # supposed to be more general than univariate.
    def __init__(self, obj, n_cat_state):
        self.obj = obj
        self.n_cat_state = n_cat_state
    
    def __call__(self, X_cat_row):
        """Return the covariance function of object evaluated at the given `X_cat_row`.
        
        Parameters
        ----------
        X_cat_row : Array-like, (2,)
            The `Xi` and `Xj` point to evaluate at.
        """
        return mpmath.chop(mpmath.diff(self.obj._mask_cov_func,
                                       X_cat_row,
                                       n=self.n_cat_state,
                                       singular=True))

MASKEDKERNEL_RESERVED_NAMES = ['base', 'mask', 'maskC', 'num_dim', 'scale']

class MaskedKernel(Kernel):
    """Creates a kernel that is only masked to operate on certain dimensions, or has scaling/shifting.
    
    This can be used, for instance, to put a squared exponential kernel in one
    direction and a Matern kernel in the other.
    
    Overrides :py:meth:`__getattribute__` and :py:meth:`__setattr__` to make all
    setting/accessing go to the `base` kernel.
    
    Parameters
    ----------
    base : :py:class:`Kernel` instance
        The :py:class:`Kernel` to apply in the dimensions specified in `mask`.
    total_dim : int, optional
        The total number of dimensions the masked kernel should have. Default
        is 2.
    mask : list or other array-like, optional
        1d list of indices of dimensions `X` to include when passing to the
        `base` kernel. Length must be `base.num_dim`. Default is [0] (i.e.,
        just pass the first column of `X` to a univariate `base` kernel).
    scale : list or other array-like, optional
        1d list of scale factors to apply to the elements in `Xi`, `Xj`. Default
        is ones. Length must be equal to 2`base.num_dim`.
    """
    def __init__(self, base, total_dim=2, mask=[0], scale=None):
        if len(mask) != base.num_dim:
            raise ValueError("Length of mask must be equal to the number of "
                             "dimensions of the base kernel!")
        if scale is None:
            scale = [1] * 2 * base.num_dim
        elif len(scale) != 2 * base.num_dim:
            raise ValueError("Length of scale must be equal to twice the number "
                             "of dimensions of the base kernel!")
        self.base = base
        super(MaskedKernel, self).__init__(num_dim=total_dim,
                                           num_params=base.num_params,
                                           initial_params=base.params,
                                           fixed_params=base.fixed_params,
                                           param_names=base.param_names,
                                           enforce_bounds=base.enforce_bounds,
                                           hyperprior=base.hyperprior)
        self.mask = mask
        # maskC is the complement of mask:
        self.maskC = range(0, self.num_dim)
        for v in self.mask:
            self.maskC.remove(v)
        self.scale = scale
    
    def __getattribute__(self, name):
        """Gets all attributes from the base kernel.
        
        The exceptions are 'base', 'mask', 'maskC', 'num_dim', 'scale' and any
        special method (i.e., a method/attribute having leading and trailing
        double underscores), which are taken from :py:class:`MaskedKernel`.
        """
        if not (name.startswith('__') and name.endswith('__')) and name not in MASKEDKERNEL_RESERVED_NAMES:
            try:
                return self.base.__getattribute__(name)
            except AttributeError:
                return super(MaskedKernel, self).__getattribute__(name)
        else:
            return super(MaskedKernel, self).__getattribute__(name)
    
    def __setattr__(self, name, value):
        """Sets all attributes in the base kernel.
        
        The exceptions are 'base', 'mask', 'maskC', 'num_dim', 'scale' and any
        special method (i.e., a method/attribute having leading and trailing
        double underscores), which are set in :py:class:`MaskedKernel`.
        """
        if not (name.startswith('__') and name.endswith('__')) and name not in MASKEDKERNEL_RESERVED_NAMES:
            return self.base.__setattr__(name, value)
        else:
            return super(MaskedKernel, self).__setattr__(name, value)
    
    def __call__(self, Xi, Xj, ni, nj, **kwargs):
        """Evaluate the covariance between points `Xi` and `Xj` with derivative order `ni`, `nj`.
        
        Note that in the argument specifications, `D` is the `total_dim`
        specified in the constructor (i.e., :py:attr:`num_dim` for the
        :py:class:`MaskedKernel` instance itself).
        
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
            (no hyperparameter derivatives).
        symmetric : bool, optional
            Whether or not the input `Xi`, `Xj` are from a symmetric matrix.
            Default is False.
        
        Returns
        -------
        Kij : :py:class:`Array`, (`M`,)
            Covariances for each of the `M` `Xi`, `Xj` pairs.
        """
        # Need to see if there are any derivatives of the masked variables:
        good_idxs = (ni[:, self.maskC] == 0).all(axis=1) & (nj[:, self.maskC] == 0).all(axis=1)
        result = scipy.zeros(Xi.shape[0])
        # Need to do the indexing kinda funny to keep the shape right:
        if good_idxs.any():
            scale_tile = scipy.tile(self.scale, (good_idxs.sum(), 1))
            result[good_idxs] = self.base(
                Xi[good_idxs][:, self.mask] * scale_tile[:, :self.base.num_dim],
                Xj[good_idxs][:, self.mask] * scale_tile[:, self.base.num_dim:],
                ni[good_idxs][:, self.mask],
                nj[good_idxs][:, self.mask],
                **kwargs
            ) * (scale_tile**scipy.hstack((ni[good_idxs][:, self.mask], nj[good_idxs][:, self.mask]))).prod(axis=1)
            # Final factor is to account for the scaling.
        return result
        