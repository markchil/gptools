# Copyright 2014 Mark Chilenski
# Author: Mark Chilenski
# Contributors: Robert McGibbon
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

"""Classes and functions to warp inputs to kernels to enable fitting of
nonstationary data. Note that this accomplishes a similar goal as the Gibbs
kernel (which is a nonstationary version of the squared exponential kernel), but
with the difference that the warpings in this module can be applied to any
existing kernel. Furthermore, whereas the Gibbs kernel implements
nonstationarity by changing the length scale of the covariance function, the
warpings in the module act by transforming the input values directly.

The module contains two core classes that work together.
:py:class:`WarpingFunction` gives you a way to wrap a given function in a way
that allows you to optimize/integrate over the hyperparameters that describe the
warping. :py:class:`WarpedKernel` is an extension of the :py:class:`Kernel` base
class and is how you apply a :py:class:`WarpingFunction` to whatever kernel you
want to warp.

Two useful warpings have been implemented at this point:
:py:class:`BetaWarpedKernel` warps the inputs using the CDF of the beta
distribution (i.e., the regularized incomplete beta function). This requires
that the starting inputs be constrained to the unit hypercube [0, 1]. In order
to get arbitrary data to this form, :py:class:`LinearWarpedKernel` allows you to
apply a linear transformation based on the known bounds of your data.

So, for example, to make a beta-warped squared exponential kernel, you simply type::
    
    k_SE = gptools.SquaredExponentialKernel()
    k_SE_beta = gptools.BetaWarpedKernel(k_SE)
    
Furthermore, if your inputs `X` are not confined to the unit hypercube [0, 1],
you should use a linear transformation to map them to it::
    
    k_SE_beta_unit = gptools.LinearWarpedKernel(k_SE_beta, X.min(axis=0), X.max(axis=0))
"""

from __future__ import division

from .core import Kernel
from ..utils import UniformJointPrior, LogNormalJointPrior, CombinedBounds, MaskedBounds

import inspect
import scipy
import scipy.special

class WarpingFunction(object):
    """Wrapper to interface a function with :py:class:`WarpedKernel`.
    
    This lets you define a simple function `fun(X, d, n, p1, p2, ...)` that
    operates on one dimension of `X` at a time and has several hyperparameters.
    
    Parameters
    ----------
    fun : callable
        Must have fingerprint `fun(X, d, n, p1, p2, ...)` where `X` is an array
        of length `M`, `d` is the index of the dimension `X` is from, `n` is a
        non-negative integer representing the order of derivative to take and
        `p1`, `p2`, ... are the parameters of the warping. Note that this form
        assumes that the warping is applied independently to each dimension.
    num_dim : positive int, optional
        Number of dimensions of the input data. Must be consistent with the `X`
        and `Xstar` values passed to the
        :py:class:`~gptools.gaussian_process.GaussianProcess` you wish to use
        the warping function with. Default is 1.
    num_params : Non-negative int, optional
        Number of parameters in the model. Default is to determine the number of
        parameters by inspection of `fun` or the other arguments provided.
    initial_params : Array, (`num_params`,), optional
        Initial values to set for the hyperparameters. Default is None, in
        which case 1 is used for the initial values.
    fixed_params : Array of bool, (`num_params`,), optional
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
        callables or :py:class:`rv_frozen` instances from :py:mod:`scipy.stats`,
        in which case a :py:class:`IndependentJointPrior` is constructed with
        these as the independent priors on each hyperparameter. Default is a
        uniform PDF on all hyperparameters.
    """
    def __init__(self, fun, num_dim=1, num_params=None, initial_params=None,
                 fixed_params=None, param_bounds=None, param_names=None,
                 enforce_bounds=False, hyperprior=None):
        self.fun = fun
        self.num_dim = num_dim
        
        # TODO: Some of this logic can probably by ported back to Kernel...
        # TODO: But, it also needs to be done better...
        if num_params is None:
            # There are three non-parameters at the start of fun's fingerprint.
            try:
                argspec = inspect.getargspec(fun)
                offset = 3
            except TypeError:
                # Need to remove self from the arg list for bound method:
                argspec = inspect.getargspec(fun.__call__)
                offset = 4
            
            if argspec[1] is None:
                self.num_params = len(argspec[0]) - offset
            else:
                # If fun uses the *args syntax, we need to get the number of
                # parameters elsewhere.
                if hyperprior is not None:
                    self.num_params = len(hyperprior.bounds)
                elif param_names is not None:
                    self.num_params = len(param_names)
                elif param_bounds is not None:
                    self.num_params = len(param_bounds)
                else:
                    raise ValueError(
                        "If warping function w uses a variable number of "
                        "arguments, you must also specify an explicit hyperprior, "
                        "list of param_names and/or list of param_bounds."
                    )
        else:
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
                raise ValueError(
                    "Must pass explicit parameter values if fixing parameters!"
                )
            initial_params = scipy.ones(self.num_params, dtype=float)
            fixed_params = scipy.zeros(self.num_params, dtype=float)
        else:
            if len(initial_params) != self.num_params:
                raise ValueError("Length of initial_params must be equal to num_params!")
            # Handle default case of fixed_params: no fixed parameters.
            if fixed_params is None:
                fixed_params = scipy.zeros(self.num_params, dtype=float)
            else:
                if len(fixed_params) != self.num_params:
                    raise ValueError("Length of fixed_params must be equal to num_params!")
        
        # Handle default case for parameter bounds -- set them all to (0, 1e16):
        if param_bounds is None:
            param_bounds = self.num_params * [(0.0, 1e16)]
        else:
            if len(param_bounds) != self.num_params:
                raise ValueError("Length of param_bounds must be equal to num_params!")
        
        # Handle default case for hyperpriors -- set them all to be uniform:
        if hyperprior is None:
            hyperprior = UniformJointPrior(param_bounds)
        else:
            try:
                iter(hyperprior)
                if len(hyperprior) != self.num_params:
                    raise ValueError(
                        "If hyperprior is a list its length must be equal to "
                        "num_params!"
                    )
                hyperprior = IndependentJointPrior(hyperprior)
            except TypeError:
                pass
        
        self.params = scipy.asarray(initial_params, dtype=float)
        self.fixed_params = scipy.asarray(fixed_params, dtype=bool)
        self.hyperprior = hyperprior
    
    def __call__(self, X, d, n):
        """Evaluate the warping function.
        
        Parameters
        ----------
        X : Array, (`M`,)
            `M` inputs corresponding to dimension `d`.
        d : non-negative int
            Index of the dimension that `X` is from.
        n : non-negative int
            Derivative order to compute.
        """
        return self.fun(X, d, n, *self.params)
    
    @property
    def param_bounds(self):
        return self.hyperprior.bounds
    
    @param_bounds.setter
    def param_bounds(self, value):
        self.hyperprior.bounds = value
    
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

def beta_cdf_warp(X, d, n, *args):
    r"""Warp inputs that are confined to the unit hypercube using the regularized incomplete beta function.
    
    Applies separately to each dimension, designed for use with
    :py:class:`WarpingFunction`.
    
    Assumes that your inputs `X` lie entirely within the unit hypercube [0, 1].
    
    Note that you may experience some issues with constraining and computing
    derivatives at :math:`x=0` when :math:`\alpha < 1` and at :math:`x=1` when
    :math:`\beta < 1`. As a workaround, try mapping your data to not touch the
    boundaries of the unit hypercube.
    
    Parameters
    ----------
    X : array, (`M`,)
        `M` inputs from dimension `d`.
    d : non-negative int
        The index (starting from zero) of the dimension to apply the warping to.
    n : non-negative int
        The derivative order to compute.
    *args : 2N scalars
        The remaining parameters to describe the warping, given as scalars.
        These are given as `alpha_i`, `beta_i` for each of the `D` dimensions.
        Note that these must ALL be provided for each call.
    
    References
    ----------
    .. [1] J. Snoek, K. Swersky, R. Zemel, R. P. Adams, "Input Warping for
       Bayesian Optimization of Non-stationary Functions" ICML (2014)
    """
    X = scipy.asarray(X)
    
    a = args[2 * d]
    b = args[2 * d + 1]
    
    if n == 0:
        return scipy.special.betainc(a, b, X)
    elif n == 1:
        # http://functions.wolfram.com/GammaBetaErf/BetaRegularized/20/01/01/
        return (1 - X)**(b - 1) * X**(a - 1) / scipy.special.beta(a, b)
    else:
        # TODO: There is a generalized form at http://functions.wolfram.com/GammaBetaErf/BetaRegularized/20/02/01/
        raise NotImplementedError(
            "Derivatives of order greater than one are not yet supported!"
        )

def linear_warp(X, d, n, *args):
    r"""Warp inputs with a linear transformation.
    
    Applies the warping
    
    .. math::
        
        w(x) = \frac{x-a}{b-a}
    
    to each dimension. If you set `a=min(X)` and `b=max(X)` then this is a
    convenient way to map your inputs to the unit hypercube.
    
    Parameters
    ----------
    X : array, (`M`,)
        `M` inputs from dimension `d`.
    d : non-negative int
        The index (starting from zero) of the dimension to apply the warping to.
    n : non-negative int
        The derivative order to compute.
    *args : 2N scalars
        The remaining parameters to describe the warping, given as scalars.
        These are given as `a_i`, `b_i` for each of the `D` dimensions. Note
        that these must ALL be provided for each call.
    """
    X = scipy.asarray(X, dtype=float)
    
    a = args[2 * d]
    b = args[2 * d + 1]
    
    if n == 0:
        return (X - a) / (b - a)
    elif n == 1:
        return 1.0 / (b - a) * scipy.ones_like(X)
    else:
        return scipy.zeros_like(X)

class WarpedKernel(Kernel):
    """Kernel which has had its inputs warped through a basic, elementwise warping function.
    
    In other words, takes :math:`k(x_1, x_2, x'_1, x'_2)` and turns it into
    :math:`k(w_1(x_1), w_2(x_2), w_1(x'_1), w_2(x'_2))`.
    """
    def __init__(self, k, w):
        self.k = k
        if not isinstance(w, WarpingFunction):
            w = WarpingFunction(w)
        self.w = w
        
        if k.num_dim != w.num_dim:
            raise ValueError("k and w must have the same number of dimensions!")
        
        self._enforce_bounds = k.enforce_bounds or w.enforce_bounds
        
        super(WarpedKernel, self).__init__(
            num_dim=k.num_dim,
            num_params=k.num_params + w.num_params,
            initial_params=scipy.concatenate((k.params, w.params)),
            fixed_params=scipy.concatenate((k.fixed_params, w.fixed_params)),
            param_names=list(k.param_names) + list(w.param_names),
            hyperprior=k.hyperprior * w.hyperprior,
            enforce_bounds=self._enforce_bounds
        )
    
    def __call__(self, Xi, Xj, ni, nj, hyper_deriv=None, symmetric=False):
        if (ni > 1).any() or (nj > 1).any():
            raise ValueError("Derivative orders greater than one are not supported!")
        wXi = scipy.zeros_like(Xi)
        wXj = scipy.zeros_like(Xj)
        for d in xrange(0, self.num_dim):
            wXi[:, d] = self.w(Xi[:, d], d, 0)
            wXj[:, d] = self.w(Xj[:, d], d, 0)
        out = self.k(wXi, wXj, ni, nj, hyper_deriv=hyper_deriv, symmetric=symmetric)
        for d in xrange(0, self.num_dim):
            first_deriv_mask_i = ni[:, d] == 1
            first_deriv_mask_j = nj[:, d] == 1
            out[first_deriv_mask_i] *= self.w(Xi[first_deriv_mask_i, d], d, 1)
            out[first_deriv_mask_j] *= self.w(Xj[first_deriv_mask_j, d], d, 1)
        return out
    
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
        self.k.enforce_bounds = v
        self.w.enforce_bounds = v
    
    @property
    def fixed_params(self):
        return CombinedBounds(self.k.fixed_params, self.w.fixed_params)
    
    @fixed_params.setter
    def fixed_params(self, value):
        value = scipy.asarray(value, dtype=bool)
        self.k.fixed_params = value[:self.k.num_params]
        self.w.fixed_params = value[self.k.num_params:self.k.num_params + self.w.num_params]
    
    @property
    def params(self):
        return CombinedBounds(self.k.params, self.w.params)
    
    @params.setter
    def params(self, value):
        value = scipy.asarray(value, dtype=float)
        self.K_up_to_date = False
        self.k.params = value[:self.k.num_params]
        self.w.params = value[self.k.num_params:self.k.num_params + self.w.num_params]
    
    @property
    def param_names(self):
        return CombinedBounds(self.k.param_names, self.w.param_names)
    
    @param_names.setter
    def param_names(self, value):
        self.k.param_names = value[:self.k.num_params]
        self.w.param_names = value[self.k.num_params:self.k.num_params + self.w.num_params]
    
    @property
    def free_params(self):
        return CombinedBounds(self.k.free_params, self.w.free_params)
    
    @free_params.setter
    def free_params(self, value):
        """Set the free parameters. Note that this bypasses enforce_bounds.
        """
        value = scipy.asarray(value, dtype=float)
        self.K_up_to_date = False
        self.k.free_params = value[:self.k.num_free_params]
        self.w.free_params = value[self.k.num_free_params:self.k.num_free_params + self.w.num_free_params]
    
    @property
    def free_param_bounds(self):
        return CombinedBounds(self.k.free_param_bounds, self.w.free_param_bounds)
    
    @free_param_bounds.setter
    def free_param_bounds(self, value):
        value = scipy.asarray(value, dtype=float)
        self.k.free_param_bounds = value[:self.k.num_free_params]
        self.w.free_param_bounds = value[self.k.num_free_params:self.k.num_free_params + self.w.num_free_params]
    
    @property
    def free_param_names(self):
        return CombinedBounds(self.k.free_param_names, self.w.free_param_names)
    
    @free_param_names.setter
    def free_param_names(self, value):
        value = scipy.asarray(value, dtype=str)
        self.K_up_to_date = False
        self.k.free_param_names = value[:self.k.num_free_params]
        self.w.free_param_names = value[self.k.num_free_params:self.k.num_free_params + self.w.num_free_params]
    
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
            num_free_k = sum(~self.k.fixed_params)
            self.k.set_hyperparams(new_params[:num_free_k])
            self.w.set_hyperparams(new_params[num_free_k:])
        else:
            raise ValueError("Length of new_params must be %s!" % (len(self.free_params),))

class BetaWarpedKernel(WarpedKernel):
    r"""Class to warp any existing :py:class:`Kernel` with the beta CDF.
    
    Assumes that your inputs `X` lie entirely within the unit hypercube [0, 1].
    
    Note that you may experience some issues with constraining and computing
    derivatives at :math:`x=0` when :math:`\alpha < 1` and at :math:`x=1` when
    :math:`\beta < 1`. As a workaround, try mapping your data to not touch the
    boundaries of the unit hypercube.
    
    Parameters
    ----------
    k : :py:class:`Kernel`
        The :py:class:`Kernel` to warp.
    **w_kwargs : optional kwargs
        All additional kwargs are passed to the constructor of
        :py:class:`WarpingFunction`. If no hyperprior or param_bounds are
        provided, takes each :math:`\alpha`, :math:`\beta` to follow the
        log-normal distribution.
    
    References
    ----------
    .. [1] J. Snoek, K. Swersky, R. Zemel, R. P. Adams, "Input Warping for
       Bayesian Optimization of Non-stationary Functions" ICML (2014)
    """
    def __init__(self, k, **w_kwargs):
        param_names = []
        for d in xrange(0, k.num_dim):
            param_names += ['\\alpha_%d' % (d,), '\\beta_%d' % (d,)]
        if 'hyperprior' not in w_kwargs and 'param_bounds' not in w_kwargs:
            w_kwargs['hyperprior'] = LogNormalJointPrior(
                [0, 0] * k.num_dim,
                [0.5, 0.5] * k.num_dim
            )
        w = WarpingFunction(
            beta_cdf_warp,
            num_dim=k.num_dim,
            param_names=param_names,
            **w_kwargs
        )
        super(BetaWarpedKernel, self).__init__(k, w)

class LinearWarpedKernel(WarpedKernel):
    """Class to warp any existing :py:class:`Kernel` with the linear transformation given in :py:func:`linear_warp`.
    
    If you set `a` to be the minimum of your `X` inputs in each dimension and `b`
    to be the maximum then you can use this to map data from an arbitrary domain
    to the unit hypercube [0, 1], as is required for application of the
    :py:class:`BetaWarpedKernel`, for instance.
    
    Parameters
    ----------
    k : :py:class:`Kernel`
        The :py:class:`Kernel` to warp.
    a : list
        The `a` parameter in the linear warping defined in :py:func:`linear_warp`.
        This list must have length equal to `k.num_dim`.
    b : list
        The `b` parameter in the linear warping defined in :py:func:`linear_warp`.
        This list must have length equal to `k.num_dim`.
    """
    def __init__(self, k, a, b):
        a = scipy.atleast_1d(scipy.asarray(a, dtype=float))
        b = scipy.atleast_1d(scipy.asarray(b, dtype=float))
        if len(a) != k.num_dim:
            raise ValueError("a must have length equal to k.num_dim!")
        if len(b) != k.num_dim:
            raise ValueError("b must have length equal to k.num_dim!")
        param_names = []
        initial_params = []
        param_bounds = []  # Set this to be narrow so the LL doesn't overflow.
        for d in xrange(0, k.num_dim):
            param_names += ['a_%d' % (d,), 'b_%d' % (d,)]
            initial_params += [a[d], b[d]]
            param_bounds += [(a[d] - 1e-3, a[d] + 1e-3), (b[d] - 1e-3, b[d] + 1e-3)]
        w = WarpingFunction(
            linear_warp,
            num_dim=k.num_dim,
            initial_params=initial_params,
            param_bounds=param_bounds,
            fixed_params=scipy.ones_like(initial_params, dtype=bool),
            param_names=param_names
        )
        super(LinearWarpedKernel, self).__init__(k, w)
