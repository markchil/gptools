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

"""Provides classes for defining explicit, parametric mean functions.

To provide the necessary hooks to optimize/sample the hyperparameters, your mean
function must be wrapped with :py:class:`MeanFunction` before being passed to
:py:class:`GaussianProcess`. The function must have the calling fingerprint
`fun(X, n, p1, p2, ...)`, where `X` is an array with shape `(M, N)`, `n` is a
vector with length `D` and `p1`, `p2`, ... are the (hyper)parameters of the mean
function, given as individual arguments.
"""

from __future__ import division

from .utils import unique_rows, UniformJointPrior, MaskedBounds

import scipy
import inspect

class MeanFunction(object):
    r"""Wrapper to turn a function into a form useable by :py:class:`GaussianProcess`.
    
    This lets you define a simple function `fun(X, n, p1, p2, ...)` that
    operates on an (`M`, `D`) array `X`, taking the derivatives indicated by the
    vector `n` with length `D` (one derivative order for each dimension). The
    function should evaluate this derivative at all points in `X`, returning an
    array of length `M`. :py:class:`MeanFunction` takes care of looping over the
    different derivatives requested by :py:class:`GaussianProcess`.
    
    Parameters
    ----------
    fun : callable
        Must have fingerprint `fun(X, n, p1, p2, ...)` where `X` is an array
        with shape (`M`, `D`), `n` is an array of non-negative integers with
        length `D` representing the order of derivative orders to take for each
        dimension and `p1`, `p2`, ... are the parameters of the mean function.
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
    def __init__(self, fun, num_params=None, initial_params=None,
                 fixed_params=None, param_bounds=None, param_names=None,
                 enforce_bounds=False, hyperprior=None):
        # TODO: This duplicates a lot of code from WarpingFunction, which itself duplicates code from Kernel. Is it worth making a wrapper?
        self.fun = fun
        
        if num_params is None:
            # There are two non-parameters at the start of fun's fingerprint.
            try:
                argspec = inspect.getargspec(fun)
                offset = 2
            except TypeError:
                # Need to remove self from the arg list for bound method:
                argspec = inspect.getargspec(fun.__call__)
                offset = 3
            
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
    
    def __call__(self, X, n):
        n = scipy.atleast_2d(scipy.asarray(n, dtype=int))
        X = scipy.atleast_2d(scipy.asarray(X))
        n_unique = unique_rows(n)
        mu = scipy.zeros(X.shape[0])
        for nn in n_unique:
            idxs = (n == nn).all(axis=1)
            mu[idxs] = self.fun(X[idxs, :], nn, *self.params)
        
        return mu
    
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

def constant(X, n, mu):
    """Function implementing a constant mean suitable for use with :py:class:`MeanFunction`.
    """
    if (n == 0).all():
        return mu * scipy.ones(X.shape[0])
    else:
        return scipy.zeros(X.shape[0])

class ConstantMeanFunction(MeanFunction):
    """Class implementing a constant mean function suitable for use with :py:class:`GaussianProcess`.
    
    All kwargs are passed to :py:class:`MeanFunction`. If you do not pass
    `hyperprior` or `param_bounds`, the hyperprior for the mean is taken to be
    uniform over [-1e3, 1e3].
    """
    def __init__(self, **kwargs):
        if 'hyperprior' not in kwargs and 'param_bounds' not in kwargs:
            kwargs['param_bounds'] = [(-1e3, 1e3)]
        super(ConstantMeanFunction, self).__init__(
            constant,
            param_names=['\\mu'],
            **kwargs
        )

# The following use the definitions from chapter 4 of JR Walk's thesis:
def mtanh(alpha, z):
    """Modified hyperbolic tangent function mtanh(z; alpha).
    
    Parameters
    ----------
    alpha : float
        The core slope of the mtanh.
    z : float or array
        The coordinate of the mtanh.
    """
    z = scipy.asarray(z)
    return ((1 + alpha * z) * scipy.exp(z) - scipy.exp(-z)) / (scipy.exp(z) + scipy.exp(-z))

def mtanh_profile(X, n, x0, delta, alpha, h, b):
    """Profile used with the mtanh function to fit profiles, suitable for use with :py:class:`MeanFunction`.
    
    Only supports univariate data!
    
    Parameters
    ----------
    X : array, (`M`, 1)
        The points to evaluate at.
    n : array, (1,)
        The order of derivative to compute. Only up to first derivatives are
        supported.
    x0 : float
        Pedestal center
    delta : float
        Pedestal halfwidth
    alpha : float
        Core slope
    h : float
        Pedestal height
    b : float
        Pedestal foot
    """
    X = X[:, 0]
    z = (x0 - X) / delta
    if n[0] == 0:
        return (h + b) / 2.0 + (h - b) * mtanh(alpha, z) / 2.0
    elif n[0] == 1:
        return -(h - b) / (2.0 * delta) * (1 + alpha / 4.0 * (1 + 2 * z + scipy.exp(2 * z))) / (scipy.cosh(z))**2
    else:
        raise NotImplementedError("Derivatives of order greater than 1 are not supported!")

class MtanhMeanFunction1d(MeanFunction):
    """Profile with mtanh edge, suitable for use with :py:class:`GaussianProcess`.
    
    All kwargs are passed to :py:class:`MeanFunction`. If `hyperprior` and
    `param_bounds` are not passed then the hyperprior is taken to be uniform
    over the following intervals:
    
        ===== ==== ===
        x0    0.98 1.1
        delta 0.0  0.1
        alpha -0.5 0.5
        h     0    5
        b     0    0.5
        ===== ==== ===
    
    """
    def __init__(self, **kwargs):
        if 'hyperprior' not in kwargs and 'param_bounds' not in kwargs:
            kwargs['param_bounds'] = [(0.98, 1.1), (0, 0.1), (-0.5, 0.5), (0, 5), (0, 0.5)]
        super(MtanhMeanFunction1d, self).__init__(
            mtanh_profile,
            param_names=['x_0', '\\delta', '\\alpha', 'h', 'b'],
            **kwargs
        )

def linear(X, n, *args):
    """Linear mean function of arbitrary dimension, suitable for use with :py:class:`MeanFunction`.
    
    The form is :math:`m_0 * X[:, 0] + m_1 * X[:, 1] + \dots + b`.
    
    Parameters
    ----------
    X : array, (`M`, `D`)
        The points to evaluate the model at.
    n : array of non-negative int, (`D`)
        The derivative order to take, specified as an integer order for each
        dimension in `X`.
    *args : num_dim+1 floats
        The slopes for each dimension, plus the constant term. Must be of the
        form `m0, m1, ..., b`.
    """
    m = scipy.asarray(args[:-1])
    b = args[-1]
    if sum(n) > 1:
        return scipy.zeros(X.shape[0])
    elif sum(n) == 0:
        return (m * X).sum(axis=1) + b
    else:
        # sum(n) == 1:
        return m[n == 1] * scipy.ones(X.shape[0])

class LinearMeanFunction(MeanFunction):
    """Linear mean function suitable for use with :py:class:`GaussianProcess`.
    
    Parameters
    ----------
    num_dim : positive int, optional
        The number of dimensions of the input data. Default is 1.
    **kwargs : optional kwargs
        All extra kwargs are passed to :py:class:`MeanFunction`. If `hyperprior`
        and `param_bounds` are not specified, all parameters are taken to have
        a uniform hyperprior over [-1e3, 1e3].
    """
    def __init__(self, num_dim=1, **kwargs):
        if 'hyperprior' not in kwargs and 'param_bounds' not in kwargs:
            kwargs['param_bounds'] = [(-1e3, 1e3)] * (num_dim + 1)
        super(LinearMeanFunction, self).__init__(
            linear,
            param_names=['m%d' % (i,) for i in range(0, num_dim)] + ['b'],
            **kwargs
        )
