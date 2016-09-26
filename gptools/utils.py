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

"""Provides convenient utilities for working with the classes and results from :py:mod:`gptools`.
"""

from __future__ import division

import collections
import warnings
import scipy
import scipy.optimize
import scipy.special
import scipy.stats
import numpy.random
import copy
import itertools
try:
    import emcee
except ImportError:
    warnings.warn(
        "Could not import emcee: MCMC sampling will not be available.",
        ImportWarning
    )
try:
    import matplotlib.pyplot as plt
    import matplotlib.widgets as mplw
    import matplotlib.gridspec as mplgs
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.patches as mplp
except ImportError:
    warnings.warn(
        "Could not import matplotlib. Plotting functions will not be available!",
        ImportWarning
    )


class JointPrior(object):
    """Abstract class for objects implementing joint priors over hyperparameters.
    
    In addition to the abstract methods defined in this template,
    implementations should also have an attribute named `bounds` which contains
    the bounds (for a prior with finite bounds) or the 95%% interval (for a
    prior which is unbounded in at least one direction).
    """
    
    def __init__(self, i=1.0):
        """Sets the interval that :py:attr:`bounds` should return.
        
        Parameters
        ----------
        i : float, optional
            The interval to return. Default is 1.0 (100%%). Another useful value
            is 95%%.
        """
        self.i = 1.0
    
    def __call__(self, theta, hyper_deriv=None):
        """Evaluate the prior log-PDF at the given values of the hyperparameters, theta.
        
        Parameters
        ----------
        theta : array-like, (`num_params`,)
            The hyperparameters to evaluate the log-PDF at.
        hyper_deriv : int or None, optional
            If present, return the derivative of the log-PDF with respect to
            the variable with this index.
        """
        raise NotImplementedError("__call__ must be implemented in your own class.")
    
    def random_draw(self, size=None):
        """Draw random samples of the hyperparameters.
        
        Parameters
        ----------
        size : None, int or array-like, optional
            The number/shape of samples to draw. If None, only one sample is
            returned. Default is None.
        """
        raise NotImplementedError("random_draw must be implemented in your own class.")
    
    def sample_u(self, q):
        r"""Extract a sample from random variates uniform on :math:`[0, 1]`.
        
        For a univariate distribution, this is simply evaluating the inverse
        CDF. To facilitate efficient sampling, this function returns a *vector*
        of PPF values, one value for each variable. Basically, the idea is that,
        given a vector :math:`q` of `num_params` values each of which is
        distributed uniformly on :math:`[0, 1]`, this function will return
        corresponding samples for each variable.
        
        Parameters
        ----------
        q : array-like, (`num_params`,)
            Values between 0 and 1 to evaluate inverse CDF at.
        """
        raise NotImplementedError("ppf must be implemented in your own class.")
    
    def elementwise_cdf(self, p):
        r"""Convert a sample to random variates uniform on :math:`[0, 1]`.
        
        For a univariate distribution, this is simply evaluating the CDF. To
        facilitate efficient sampling, this function returns a *vector* of CDF
        values, one value for each variable. Basically, the idea is that, given
        a vector :math:`q` of `num_params` values each of which is distributed
        according to the prior, this function will return variables uniform on
        :math:`[0, 1]` corresponding to each variable. This is the inverse
        operation to :py:meth:`sample_u`.
        
        Parameters
        ----------
        p : array-like, (`num_params`,)
            Values to evaluate CDF at.
        """
        raise NotImplementedError("cdf must be implemented in your own class.")
    
    def __mul__(self, other):
        """Multiply two :py:class:`JointPrior` instances together.
        """
        return ProductJointPrior(self, other)

class CombinedBounds(object):
    """Object to support reassignment of the bounds from a combined prior.
    
    Works for any types of arrays.
    
    Parameters
    ----------
    l1 : array-like
        The first list.
    l2 : array-like
        The second list.
    """
    # TODO: This could use a lot more work!
    def __init__(self, l1, l2):
        self.l1 = l1
        self.l2 = l2
    
    def __getitem__(self, pos):
        """Get the item(s) at `pos`.
        
        `pos` can be a basic slice object. But, the method is implemented by
        turning the internal array-like objects into lists, so only the basic
        indexing capabilities supported by the list data type can be used.
        """
        return (list(self.l1) + list(self.l2))[pos]
    
    def __setitem__(self, pos, value):
        """Set the item at location pos to value.
        
        Only works for scalar indices.
        """
        if pos < len(self.l1):
            self.l1[pos] = value
        else:
            self.l2[pos - len(self.l1)] = value
    
    def __len__(self):
        """Get the length of the combined arrays.
        """
        return len(self.l1) + len(self.l2)
    
    def __invert__(self):
        """Return the elementwise inverse.
        """
        return ~scipy.asarray(self)
    
    def __str__(self):
        """Get user-friendly string representation.
        """
        return str(self[:])
    
    def __repr__(self):
        """Get exact string representation.
        """
        return str(self) + " from CombinedBounds(" + str(self.l1) + ", " + str(self.l2) + ")"

class MaskedBounds(object):
    """Object to support reassignment of free parameter bounds.
    
    Parameters
    ----------
    a : array
        The array to be masked.
    m : array of int
        The indices in `a` which are to be accessible.
    """
    def __init__(self, a, m):
        self.a = a
        self.m = m
    
    def __getitem__(self, pos):
        """Get the item(s) at location `pos` in the masked array.
        """
        return self.a[self.m[pos]]
    
    def __setitem__(self, pos, value):
        """Set the item(s) at location `pos` in the masked array.
        """
        self.a[self.m[pos]] = value
    
    def __len__(self):
        """Get the length of the masked array.
        """
        return len(self.m)
    
    def __str__(self):
        """Get user-friendly string representation.
        """
        return str(self[:])
    
    def __repr__(self):
        """Get exact string representation.
        """
        return str(self) + " from MaskedBounds(" + str(self.a) + ", " + str(self.m) + ")"

class ProductJointPrior(JointPrior):
    """Product of two independent priors.
    
    Parameters
    ----------
    p1, p2: :py:class:`JointPrior` instances
        The two priors to merge.
    """
    def __init__(self, p1, p2):
        if not isinstance(p1, JointPrior) or not isinstance(p2, JointPrior):
            raise TypeError(
                "Both arguments to ProductPrior must be instances of JointPrior!"
            )
        self.p1 = p1
        self.p2 = p2
    
    @property
    def i(self):
        return min(self.p1.i, self.p2.i)
    
    @i.setter
    def i(self, v):
        self.p1.i = i
        self.p2.i = i
    
    @property
    def bounds(self):
        return CombinedBounds(self.p1.bounds, self.p2.bounds)
    
    @bounds.setter
    def bounds(self, v):
        num_p1_bounds = len(self.p1.bounds)
        self.p1.bounds = v[:num_p1_bounds]
        self.p2.bounds = v[num_p1_bounds:]

    def __call__(self, theta, hyper_deriv=None):
        """Evaluate the prior log-PDF at the given values of the hyperparameters, theta.
        
        The log-PDFs of the two priors are summed.
        
        Parameters
        ----------
        theta : array-like, (`num_params`,)
            The hyperparameters to evaluate the log-PDF at.
        """
        p1_num_params = len(self.p1.bounds)
        if hyper_deriv is not None:
            if hyper_deriv < p1_num_params:
                return self.p1(theta[:p1_num_params], hyper_deriv=hyper_deriv)
            else:
                return self.p2(theta[p1_num_params:], hyper_deriv=hyper_deriv - p1_num_params)
        return self.p1(theta[:p1_num_params]) + self.p2(theta[p1_num_params:])
    
    def sample_u(self, q):
        r"""Extract a sample from random variates uniform on :math:`[0, 1]`.
        
        For a univariate distribution, this is simply evaluating the inverse
        CDF. To facilitate efficient sampling, this function returns a *vector*
        of PPF values, one value for each variable. Basically, the idea is that,
        given a vector :math:`q` of `num_params` values each of which is
        distributed uniformly on :math:`[0, 1]`, this function will return
        corresponding samples for each variable.
        
        Parameters
        ----------
        q : array-like, (`num_params`,)
            Values between 0 and 1 to evaluate inverse CDF at.
        """
        p1_num_params = len(self.p1.bounds)
        return scipy.concatenate(
            (
                self.p1.sample_u(q[:p1_num_params]),
                self.p2.sample_u(q[p1_num_params:])
            )
        )
    
    def elementwise_cdf(self, p):
        r"""Convert a sample to random variates uniform on :math:`[0, 1]`.
        
        For a univariate distribution, this is simply evaluating the CDF. To
        facilitate efficient sampling, this function returns a *vector* of CDF
        values, one value for each variable. Basically, the idea is that, given
        a vector :math:`q` of `num_params` values each of which is distributed
        according to the prior, this function will return variables uniform on
        :math:`[0, 1]` corresponding to each variable. This is the inverse
        operation to :py:meth:`sample_u`.
        
        Parameters
        ----------
        p : array-like, (`num_params`,)
            Values to evaluate CDF at.
        """
        p1_num_params = len(self.p1.bounds)
        return scipy.concatenate(
            (
                self.p1.elementwise_cdf(p[:p1_num_params]),
                self.p2.elementwise_cdf(p[p1_num_params:])
            )
        )
    
    def random_draw(self, size=None):
        """Draw random samples of the hyperparameters.
        
        The outputs of the two priors are stacked vertically.
        
        Parameters
        ----------
        size : None, int or array-like, optional
            The number/shape of samples to draw. If None, only one sample is
            returned. Default is None.
        """
        draw_1 = self.p1.random_draw(size=size)
        draw_2 = self.p2.random_draw(size=size)
        
        if draw_1.ndim == 1:
            return scipy.hstack((draw_1, draw_2))
        else:
            return scipy.vstack((draw_1, draw_2))

class UniformJointPrior(JointPrior):
    """Uniform prior over the specified bounds.
    
    Parameters
    ----------
    bounds : list of tuples, (`num_params`,)
        The bounds for each of the random variables.
    ub : list of float, (`num_params`,), optional
        The upper bounds for each of the random variables. If present, `bounds`
        is then taken to be a list of float with the lower bounds. This gives
        :py:class:`UniformJointPrior` a similar calling fingerprint as the other
        :py:class:`JointPrior` classes.
    """
    def __init__(self, bounds, ub=None, **kwargs):
        super(UniformJointPrior, self).__init__(**kwargs)
        if ub is not None:
            try:
                bounds = zip(bounds, ub)
            except TypeError:
                bounds = [(bounds, ub)]
        self.bounds = bounds
    
    def __call__(self, theta, hyper_deriv=None):
        """Evaluate the prior log-PDF at the given values of the hyperparameters, theta.
        
        Parameters
        ----------
        theta : array-like, (`num_params`,)
            The hyperparameters to evaluate the log-PDF at.
        """
        if hyper_deriv is not None:
            return 0.0
        ll = 0.0
        for v, b in zip(theta, self.bounds):
            if b[0] <= v and v <= b[1]:
                ll += -scipy.log(b[1] - b[0])
            else:
                ll = -scipy.inf
                break
        return ll
    
    def sample_u(self, q):
        r"""Extract a sample from random variates uniform on :math:`[0, 1]`.
        
        For a univariate distribution, this is simply evaluating the inverse
        CDF. To facilitate efficient sampling, this function returns a *vector*
        of PPF values, one value for each variable. Basically, the idea is that,
        given a vector :math:`q` of `num_params` values each of which is
        distributed uniformly on :math:`[0, 1]`, this function will return
        corresponding samples for each variable.
        
        Parameters
        ----------
        q : array of float
            Values between 0 and 1 to evaluate inverse CDF at.
        """
        q = scipy.atleast_1d(q)
        if len(q) != len(self.bounds):
            raise ValueError("length of q must equal the number of parameters!")
        if q.ndim != 1:
            raise ValueError("q must be one-dimensional!")
        if (q < 0).any() or (q > 1).any():
            raise ValueError("q must be within [0, 1]!")
        return scipy.asarray([(b[1] - b[0]) * v + b[0] for v, b in zip(q, self.bounds)])
    
    def elementwise_cdf(self, p):
        r"""Convert a sample to random variates uniform on :math:`[0, 1]`.
        
        For a univariate distribution, this is simply evaluating the CDF. To
        facilitate efficient sampling, this function returns a *vector* of CDF
        values, one value for each variable. Basically, the idea is that, given
        a vector :math:`q` of `num_params` values each of which is distributed
        according to the prior, this function will return variables uniform on
        :math:`[0, 1]` corresponding to each variable. This is the inverse
        operation to :py:meth:`sample_u`.
        
        Parameters
        ----------
        p : array-like, (`num_params`,)
            Values to evaluate CDF at.
        """
        p = scipy.atleast_1d(p)
        if len(p) != len(self.bounds):
            raise ValueError("length of p must equal the number of parameters!")
        if p.ndim != 1:
            raise ValueError("p must be one-dimensional!")
        c = scipy.zeros(len(self.bounds))
        for k in xrange(0, len(self.bounds)):
            if p[k] <= self.bounds[k][0]:
                c[k] = 0.0
            elif p[k] >= self.bounds[k][1]:
                c[k] = 1.0
            else:
                c[k] = (p[k] - self.bounds[k][0]) / (self.bounds[k][1] - self.bounds[k][0])
        return c
    
    def random_draw(self, size=None):
        """Draw random samples of the hyperparameters.
        
        Parameters
        ----------
        size : None, int or array-like, optional
            The number/shape of samples to draw. If None, only one sample is
            returned. Default is None.
        """
        return scipy.asarray([numpy.random.uniform(low=b[0], high=b[1], size=size) for b in self.bounds])

class CoreEdgeJointPrior(UniformJointPrior):
    """Prior for use with Gibbs kernel warping functions with an inequality constraint between the core and edge length scales.
    """
    
    def __call__(self, theta, hyper_deriv=None):
        """Evaluate the prior log-PDF at the given values of the hyperparameters, theta.
        
        Parameters
        ----------
        theta : array-like, (`num_params`,)
            The hyperparameters to evaluate the log-PDF at.
        """
        if hyper_deriv is not None:
            return 0.0
        ll = 0
        bounds_new = copy.copy(self.bounds)
        bounds_new[2] = (self.bounds[2][0], theta[1])
        for v, b in zip(theta, bounds_new):
            if b[0] <= v and v <= b[1]:
                ll += -scipy.log(b[1] - b[0])
            else:
                ll = -scipy.inf
                break
        return ll
    
    def sample_u(self, q):
        r"""Extract a sample from random variates uniform on :math:`[0, 1]`.
        
        For a univariate distribution, this is simply evaluating the inverse
        CDF. To facilitate efficient sampling, this function returns a *vector*
        of PPF values, one value for each variable. Basically, the idea is that,
        given a vector :math:`q` of `num_params` values each of which is
        distributed uniformly on :math:`[0, 1]`, this function will return
        corresponding samples for each variable.
        
        Parameters
        ----------
        q : array of float
            Values between 0 and 1 to evaluate inverse CDF at.
        """
        # TODO: Do this!
        raise NotImplementedError("Not done yet!")
    
    def elementwise_cdf(self, p):
        r"""Convert a sample to random variates uniform on :math:`[0, 1]`.
        
        For a univariate distribution, this is simply evaluating the CDF. To
        facilitate efficient sampling, this function returns a *vector* of CDF
        values, one value for each variable. Basically, the idea is that, given
        a vector :math:`q` of `num_params` values each of which is distributed
        according to the prior, this function will return variables uniform on
        :math:`[0, 1]` corresponding to each variable. This is the inverse
        operation to :py:meth:`sample_u`.
        
        Parameters
        ----------
        p : array-like, (`num_params`,)
            Values to evaluate CDF at.
        """
        # TODO: Do this!
        raise NotImplementedError("Not done yet!")
    
    def random_draw(self, size=None):
        """Draw random samples of the hyperparameters.
        
        Parameters
        ----------
        size : None, int or array-like, optional
            The number/shape of samples to draw. If None, only one sample is
            returned. Default is None.
        """
        if size is None:
            size = 1
            single_val = True
        else:
            single_val = False
        
        out_shape = [len(self.bounds)]
        try:
            out_shape.extend(size)
        except TypeError:
            out_shape.append(size)
        
        out = scipy.zeros(out_shape)
        for j in xrange(0, len(self.bounds)):
            if j != 2:
                out[j, :] = numpy.random.uniform(low=self.bounds[j][0],
                                                 high=self.bounds[j][1],
                                                 size=size)
            else:
                out[j, :] = numpy.random.uniform(low=self.bounds[j][0],
                                                 high=out[j - 1, :],
                                                 size=size)
        if not single_val:
            return out
        else:
            return out.ravel()

class CoreMidEdgeJointPrior(UniformJointPrior):
    """Prior for use with Gibbs kernel warping functions with an inequality constraint between the core, mid and edge length scales and the core-mid and mid-edge joins.
    """
    
    def __call__(self, theta, hyper_deriv=None):
        """Evaluate the prior log-PDF at the given values of the hyperparameters, theta.
        
        Parameters
        ----------
        theta : array-like, (`num_params`,)
            The hyperparameters to evaluate the log-PDF at.
        """
        if hyper_deriv is not None:
            return 0.0
        ll = 0
        bounds_new = copy.copy(self.bounds)
        # lc < lm:
        # bounds_new[1] = (self.bounds[1][0], theta[2])
        # le < lm:
        # bounds_new[3] = (self.bounds[3][0], theta[2])
        # xa < xb:
        bounds_new[6] = (self.bounds[6][0], theta[7])
        for v, b in zip(theta, bounds_new):
            if b[0] <= v and v <= b[1]:
                ll += -scipy.log(b[1] - b[0])
            else:
                ll = -scipy.inf
                break
        return ll
    
    def sample_u(self, q):
        r"""Extract a sample from random variates uniform on :math:`[0, 1]`.
        
        For a univariate distribution, this is simply evaluating the inverse
        CDF. To facilitate efficient sampling, this function returns a *vector*
        of PPF values, one value for each variable. Basically, the idea is that,
        given a vector :math:`q` of `num_params` values each of which is
        distributed uniformly on :math:`[0, 1]`, this function will return
        corresponding samples for each variable.
        
        Parameters
        ----------
        q : array of float
            Values between 0 and 1 to evaluate inverse CDF at.
        """
        # TODO: Do this!
        raise NotImplementedError("Not done yet!")
    
    def elementwise_cdf(self, p):
        r"""Convert a sample to random variates uniform on :math:`[0, 1]`.
        
        For a univariate distribution, this is simply evaluating the CDF. To
        facilitate efficient sampling, this function returns a *vector* of CDF
        values, one value for each variable. Basically, the idea is that, given
        a vector :math:`q` of `num_params` values each of which is distributed
        according to the prior, this function will return variables uniform on
        :math:`[0, 1]` corresponding to each variable. This is the inverse
        operation to :py:meth:`sample_u`.
        
        Parameters
        ----------
        p : array-like, (`num_params`,)
            Values to evaluate CDF at.
        """
        # TODO: Do this!
        raise NotImplementedError("Not done yet!")
    
    def random_draw(self, size=None):
        """Draw random samples of the hyperparameters.
        
        Parameters
        ----------
        size : None, int or array-like, optional
            The number/shape of samples to draw. If None, only one sample is
            returned. Default is None.
        """
        if size is None:
            size = 1
            single_val = True
        else:
            single_val = False
        
        out_shape = [len(self.bounds)]
        try:
            out_shape.extend(size)
        except TypeError:
            out_shape.append(size)
        
        out = scipy.zeros(out_shape)
        # sigma_f, lm, la, lb, xb:
        for j in [0, 1, 2, 3, 4, 5, 7]:
            out[j, :] = numpy.random.uniform(low=self.bounds[j][0],
                                             high=self.bounds[j][1],
                                             size=size)
        # lc, le:
        # for j in [1, 3]:
        #     out[j, :] = numpy.random.uniform(low=self.bounds[j][0],
        #                                      high=out[2, :],
        #                                      size=size)
        # xa:
        out[6, :] = numpy.random.uniform(low=self.bounds[6][0],
                                         high=out[7, :],
                                         size=size)
        if not single_val:
            return out
        else:
            return out.ravel()

class IndependentJointPrior(JointPrior):
    """Joint prior for which each hyperparameter is independent.
    
    Parameters
    ----------
    univariate_priors : list of callables or rv_frozen, (`num_params`,)
        The univariate priors for each hyperparameter. Entries in this list
        can either be a callable that takes as an argument the entire list of
        hyperparameters or a frozen instance of a distribution from
        :py:mod:`scipy.stats`.
    """
    def __init__(self, univariate_priors):
        super(IndependentJointPrior, self).__init__(**kwargs)
        self.univariate_priors = univariate_priors
    
    def __call__(self, theta, hyper_deriv=None):
        """Evaluate the prior log-PDF at the given values of the hyperparameters, theta.
        
        Parameters
        ----------
        theta : array-like, (`num_params`,)
            The hyperparameters to evaluate the log-PDF at.
        """
        if hyper_deriv is not None:
            raise NotImplementedError(
                "Hyperparameter derivatives not supported for IndependentJointPrior!"
            )
        ll = 0
        for v, p in zip(theta, self.univariate_priors):
            try:
                ll += p(theta)
            except TypeError:
                ll += p.logpdf(v)
        return ll
    
    @property
    def bounds(self):
        """The bounds of the random variable.
        
        Set `self.i=0.95` to return the 95% interval if this is used for setting
        bounds on optimizers/etc. where infinite bounds may not be useful.
        """
        return [p.interval(self.i) for p in self.univariate_priors]
    
    def sample_u(self, q):
        r"""Extract a sample from random variates uniform on :math:`[0, 1]`.
        
        For a univariate distribution, this is simply evaluating the inverse
        CDF. To facilitate efficient sampling, this function returns a *vector*
        of PPF values, one value for each variable. Basically, the idea is that,
        given a vector :math:`q` of `num_params` values each of which is
        distributed uniformly on :math:`[0, 1]`, this function will return
        corresponding samples for each variable.
        
        Parameters
        ----------
        q : array of float
            Values between 0 and 1 to evaluate inverse CDF at.
        """
        q = scipy.atleast_1d(q)
        if len(q) != len(self.univariate_priors):
            raise ValueError("length of q must equal the number of parameters!")
        if q.ndim != 1:
            raise ValueError("q must be one-dimensional!")
        if (q < 0).any() or (q > 1).any():
            raise ValueError("q must be within [0, 1]!")
        return scipy.asarray([p.ppf(v) for v, p in zip(q, self.univariate_priors)])
    
    def elementwise_cdf(self, p):
        r"""Convert a sample to random variates uniform on :math:`[0, 1]`.
        
        For a univariate distribution, this is simply evaluating the CDF. To
        facilitate efficient sampling, this function returns a *vector* of CDF
        values, one value for each variable. Basically, the idea is that, given
        a vector :math:`q` of `num_params` values each of which is distributed
        according to the prior, this function will return variables uniform on
        :math:`[0, 1]` corresponding to each variable. This is the inverse
        operation to :py:meth:`sample_u`.
        
        Parameters
        ----------
        p : array-like, (`num_params`,)
            Values to evaluate CDF at.
        """
        p = scipy.atleast_1d(p)
        if len(p) != len(self.univariate_priors):
            raise ValueError("length of p must equal the number of parameters!")
        if p.ndim != 1:
            raise ValueError("p must be one-dimensional!")
        return scipy.asarray([pr.cdf(v) for v, pr in zip(p, self.univariate_priors)])
    
    def random_draw(self, size=None):
        """Draw random samples of the hyperparameters.
        
        Parameters
        ----------
        size : None, int or array-like, optional
            The number/shape of samples to draw. If None, only one sample is
            returned. Default is None.
        """
        return scipy.asarray([p.rvs(size=size) for p in self.univariate_priors])

class NormalJointPrior(JointPrior):
    """Joint prior for which each hyperparameter has a normal prior with fixed hyper-hyperparameters.
    
    Parameters
    ----------
    mu : list of float, same size as `sigma`
        Means of the hyperparameters.
    sigma : list of float
        Standard deviations of the hyperparameters.
    """
    def __init__(self, mu, sigma, **kwargs):
        super(NormalJointPrior, self).__init__(**kwargs)
        sigma = scipy.atleast_1d(scipy.asarray(sigma, dtype=float))
        mu = scipy.atleast_1d(scipy.asarray(mu, dtype=float))
        if sigma.shape != mu.shape:
            raise ValueError("sigma and mu must have the same shape!")
        if sigma.ndim != 1:
            raise ValueError("sigma and mu must both be one dimensional!")
        self.sigma = sigma
        self.mu = mu
    
    def __call__(self, theta, hyper_deriv=None):
        """Evaluate the prior log-PDF at the given values of the hyperparameters, theta.
        
        Parameters
        ----------
        theta : array-like, (`num_params`,)
            The hyperparameters to evaluate the log-PDF at.
        """
        if hyper_deriv is not None:
            return (self.mu[hyper_deriv] - theta[hyper_deriv]) / self.sigma[hyper_deriv]**2.0
        ll = 0
        for v, s, m in zip(theta, self.sigma, self.mu):
            ll += scipy.stats.norm.logpdf(v, loc=m, scale=s)
        return ll
    
    @property
    def bounds(self):
        """The bounds of the random variable.
        
        Set `self.i=0.95` to return the 95% interval if this is used for setting
        bounds on optimizers/etc. where infinite bounds may not be useful.
        """
        return [scipy.stats.norm.interval(self.i, loc=m, scale=s) for s, m in zip(self.sigma, self.mu)]
    
    def sample_u(self, q):
        r"""Extract a sample from random variates uniform on :math:`[0, 1]`.
        
        For a univariate distribution, this is simply evaluating the inverse
        CDF. To facilitate efficient sampling, this function returns a *vector*
        of PPF values, one value for each variable. Basically, the idea is that,
        given a vector :math:`q` of `num_params` values each of which is
        distributed uniformly on :math:`[0, 1]`, this function will return
        corresponding samples for each variable.
        
        Parameters
        ----------
        q : array of float
            Values between 0 and 1 to evaluate inverse CDF at.
        """
        q = scipy.atleast_1d(q)
        if len(q) != len(self.sigma):
            raise ValueError("length of q must equal the number of parameters!")
        if q.ndim != 1:
            raise ValueError("q must be one-dimensional!")
        if (q < 0).any() or (q > 1).any():
            raise ValueError("q must be within [0, 1]!")
        return scipy.asarray([scipy.stats.norm.ppf(v, loc=m, scale=s) for v, s, m in zip(q, self.sigma, self.mu)])
    
    def elementwise_cdf(self, p):
        r"""Convert a sample to random variates uniform on :math:`[0, 1]`.
        
        For a univariate distribution, this is simply evaluating the CDF. To
        facilitate efficient sampling, this function returns a *vector* of CDF
        values, one value for each variable. Basically, the idea is that, given
        a vector :math:`q` of `num_params` values each of which is distributed
        according to the prior, this function will return variables uniform on
        :math:`[0, 1]` corresponding to each variable. This is the inverse
        operation to :py:meth:`sample_u`.
        
        Parameters
        ----------
        p : array-like, (`num_params`,)
            Values to evaluate CDF at.
        """
        p = scipy.atleast_1d(p)
        if len(p) != len(self.sigma):
            raise ValueError("length of p must equal the number of parameters!")
        if p.ndim != 1:
            raise ValueError("p must be one-dimensional!")
        return scipy.asarray([scipy.stats.norm.cdf(v, loc=m, scale=s) for v, s, m in zip(p, self.sigma, self.mu)])
    
    def random_draw(self, size=None):
        """Draw random samples of the hyperparameters.
        
        Parameters
        ----------
        size : None, int or array-like, optional
            The number/shape of samples to draw. If None, only one sample is
            returned. Default is None.
        """
        return scipy.asarray([scipy.stats.norm.rvs(loc=m, scale=s, size=size) for s, m in zip(self.sigma, self.mu)])

class LogNormalJointPrior(JointPrior):
    """Joint prior for which each hyperparameter has a log-normal prior with fixed hyper-hyperparameters.
    
    Parameters
    ----------
    mu : list of float, same size as `sigma`
        Means of the logarithms of the hyperparameters.
    sigma : list of float
        Standard deviations of the logarithms of the hyperparameters.
    """
    def __init__(self, mu, sigma, **kwargs):
        super(LogNormalJointPrior, self).__init__(**kwargs)
        sigma = scipy.atleast_1d(scipy.asarray(sigma, dtype=float))
        mu = scipy.atleast_1d(scipy.asarray(mu, dtype=float))
        if sigma.shape != mu.shape:
            raise ValueError("sigma and mu must have the same shape!")
        if sigma.ndim != 1:
            raise ValueError("sigma and mu must both be one dimensional!")
        self.sigma = sigma
        self.emu = scipy.exp(mu)
    
    def __call__(self, theta, hyper_deriv=None):
        """Evaluate the prior log-PDF at the given values of the hyperparameters, theta.
        
        Parameters
        ----------
        theta : array-like, (`num_params`,)
            The hyperparameters to evaluate the log-PDF at.
        """
        if hyper_deriv is not None:
            return -1.0 / theta[hyper_deriv] * (
                1.0 + scipy.log(theta[hyper_deriv] / self.emu[hyper_deriv]) /
                self.sigma[hyper_deriv]**2.0
            )
        ll = 0
        for v, s, em in zip(theta, self.sigma, self.emu):
            ll += scipy.stats.lognorm.logpdf(v, s, loc=0, scale=em)
        return ll
    
    @property
    def bounds(self):
        """The bounds of the random variable.
        
        Set `self.i=0.95` to return the 95% interval if this is used for setting
        bounds on optimizers/etc. where infinite bounds may not be useful.
        """
        return [scipy.stats.lognorm.interval(self.i, s, loc=0, scale=em) for s, em in zip(self.sigma, self.emu)]
    
    def sample_u(self, q):
        r"""Extract a sample from random variates uniform on :math:`[0, 1]`.
        
        For a univariate distribution, this is simply evaluating the inverse
        CDF. To facilitate efficient sampling, this function returns a *vector*
        of PPF values, one value for each variable. Basically, the idea is that,
        given a vector :math:`q` of `num_params` values each of which is
        distributed uniformly on :math:`[0, 1]`, this function will return
        corresponding samples for each variable.
        
        Parameters
        ----------
        q : array of float
            Values between 0 and 1 to evaluate inverse CDF at.
        """
        q = scipy.atleast_1d(q)
        if len(q) != len(self.sigma):
            raise ValueError("length of q must equal the number of parameters!")
        if q.ndim != 1:
            raise ValueError("q must be one-dimensional!")
        if (q < 0).any() or (q > 1).any():
            raise ValueError("q must be within [0, 1]!")
        return scipy.asarray([scipy.stats.lognorm.ppf(v, s, loc=0, scale=em) for v, s, em in zip(q, self.sigma, self.emu)])
    
    def elementwise_cdf(self, p):
        r"""Convert a sample to random variates uniform on :math:`[0, 1]`.
        
        For a univariate distribution, this is simply evaluating the CDF. To
        facilitate efficient sampling, this function returns a *vector* of CDF
        values, one value for each variable. Basically, the idea is that, given
        a vector :math:`q` of `num_params` values each of which is distributed
        according to the prior, this function will return variables uniform on
        :math:`[0, 1]` corresponding to each variable. This is the inverse
        operation to :py:meth:`sample_u`.
        
        Parameters
        ----------
        p : array-like, (`num_params`,)
            Values to evaluate CDF at.
        """
        p = scipy.atleast_1d(p)
        if len(p) != len(self.sigma):
            raise ValueError("length of p must equal the number of parameters!")
        if p.ndim != 1:
            raise ValueError("p must be one-dimensional!")
        return scipy.asarray([scipy.stats.lognorm.cdf(v, s, loc=0, scale=em) for v, s, em in zip(p, self.sigma, self.emu)])
    
    def random_draw(self, size=None):
        """Draw random samples of the hyperparameters.
        
        Parameters
        ----------
        size : None, int or array-like, optional
            The number/shape of samples to draw. If None, only one sample is
            returned. Default is None.
        """
        return scipy.asarray([scipy.stats.lognorm.rvs(s, loc=0, scale=em, size=size) for s, em in zip(self.sigma, self.emu)])

class GammaJointPrior(JointPrior):
    """Joint prior for which each hyperparameter has a gamma prior with fixed hyper-hyperparameters.
    
    Parameters
    ----------
    a : list of float, same size as `b`
        Shape parameters.
    b : list of float
        Rate parameters.
    """
    def __init__(self, a, b, **kwargs):
        super(GammaJointPrior, self).__init__(**kwargs)
        a = scipy.atleast_1d(scipy.asarray(a, dtype=float))
        b = scipy.atleast_1d(scipy.asarray(b, dtype=float))
        if a.shape != b.shape:
            raise ValueError("a and b must have the same shape!")
        if a.ndim != 1:
            raise ValueError("a and b must both be one dimensional!")
        self.a = a
        self.b = b
    
    def __call__(self, theta, hyper_deriv=None):
        """Evaluate the prior log-PDF at the given values of the hyperparameters, theta.
        
        Parameters
        ----------
        theta : array-like, (`num_params`,)
            The hyperparameters to evaluate the log-PDF at.
        """
        if hyper_deriv is not None:
            if self.a[hyper_deriv] == 1.0 and theta[hyper_deriv] == 0.0:
                return -self.b[hyper_deriv]
            else:
                return (self.a[hyper_deriv] - 1.0) / theta[hyper_deriv] - self.b[hyper_deriv]
        ll = 0
        for v, a, b in zip(theta, self.a, self.b):
            ll += scipy.stats.gamma.logpdf(v, a, loc=0, scale=1.0 / b)
        return ll
    
    @property
    def bounds(self):
        """The bounds of the random variable.
        
        Set `self.i=0.95` to return the 95% interval if this is used for setting
        bounds on optimizers/etc. where infinite bounds may not be useful.
        """
        return [scipy.stats.gamma.interval(self.i, a, loc=0, scale=1.0 / b) for a, b in zip(self.a, self.b)]
    
    def sample_u(self, q):
        r"""Extract a sample from random variates uniform on :math:`[0, 1]`.
        
        For a univariate distribution, this is simply evaluating the inverse
        CDF. To facilitate efficient sampling, this function returns a *vector*
        of PPF values, one value for each variable. Basically, the idea is that,
        given a vector :math:`q` of `num_params` values each of which is
        distributed uniformly on :math:`[0, 1]`, this function will return
        corresponding samples for each variable.
        
        Parameters
        ----------
        q : array of float
            Values between 0 and 1 to evaluate inverse CDF at.
        """
        q = scipy.atleast_1d(q)
        if len(q) != len(self.a):
            raise ValueError("length of q must equal the number of parameters!")
        if q.ndim != 1:
            raise ValueError("q must be one-dimensional!")
        if (q < 0).any() or (q > 1).any():
            raise ValueError("q must be within [0, 1]!")
        return scipy.asarray([scipy.stats.gamma.ppf(v, a, loc=0, scale=1.0 / b) for v, a, b in zip(q, self.a, self.b)])
    
    def elementwise_cdf(self, p):
        r"""Convert a sample to random variates uniform on :math:`[0, 1]`.
        
        For a univariate distribution, this is simply evaluating the CDF. To
        facilitate efficient sampling, this function returns a *vector* of CDF
        values, one value for each variable. Basically, the idea is that, given
        a vector :math:`q` of `num_params` values each of which is distributed
        according to the prior, this function will return variables uniform on
        :math:`[0, 1]` corresponding to each variable. This is the inverse
        operation to :py:meth:`sample_u`.
        
        Parameters
        ----------
        p : array-like, (`num_params`,)
            Values to evaluate CDF at.
        """
        p = scipy.atleast_1d(p)
        if len(p) != len(self.a):
            raise ValueError("length of p must equal the number of parameters!")
        if p.ndim != 1:
            raise ValueError("p must be one-dimensional!")
        return scipy.asarray([scipy.stats.gamma.cdf(v, a, loc=0, scale=1.0 / b) for v, a, b in zip(p, self.a, self.b)])
    
    def random_draw(self, size=None):
        """Draw random samples of the hyperparameters.
        
        Parameters
        ----------
        size : None, int or array-like, optional
            The number/shape of samples to draw. If None, only one sample is
            returned. Default is None.
        """
        return scipy.asarray([scipy.stats.gamma.rvs(a, loc=0, scale=1.0 / b, size=size) for a, b in zip(self.a, self.b)])

class GammaJointPriorAlt(GammaJointPrior):
    """Joint prior for which each hyperparameter has a gamma prior with fixed hyper-hyperparameters.
    
    This is an alternate form that lets you specify the mode and standard
    deviation instead of the shape and rate parameters.
    
    Parameters
    ----------
    m : list of float, same size as `s`
        Modes
    s : list of float
        Standard deviations
    """
    def __init__(self, m, s, i=1.0):
        self.i = i
        m = scipy.atleast_1d(scipy.asarray(m, dtype=float))
        s = scipy.atleast_1d(scipy.asarray(s, dtype=float))
        if m.shape != s.shape:
            raise ValueError("s and mu must have the same shape!")
        if m.ndim != 1:
            raise ValueError("s and mu must both be one dimensional!")
        self.m = m
        self.s = s
    
    @property
    def a(self):
        return 1.0 + self.b * self.m
    
    @property
    def b(self):
        return (self.m + scipy.sqrt(self.m**2 + 4.0 * self.s**2)) / (2.0 * self.s**2)

class SortedUniformJointPrior(JointPrior):
    """Joint prior for a set of variables which must be strictly increasing but are otherwise uniformly-distributed.
    
    Parameters
    ----------
    num_var : int
        The number of variables represented.
    lb : float
        The lower bound for all of the variables.
    ub : float
        The upper bound for all of the variables.
    """
    def __init__(self, num_var, lb, ub, **kwargs):
        super(SortedUniformJointPrior, self).__init__(**kwargs)
        self.num_var = num_var
        self.lb = lb
        self.ub = ub
    
    def __call__(self, theta, hyper_deriv=None):
        """Evaluate the log-probability of the variables.
        
        Parameters
        ----------
        theta : array
            The parameters to find the log-probability of.
        """
        if hyper_deriv is not None:
            return 0.0
        theta = scipy.asarray(theta)
        if (scipy.sort(theta) != theta).all() or (theta < self.lb).any() or (theta > self.ub).any():
            return -scipy.inf
        else:
            return (
                scipy.log(scipy.misc.factorial(self.num_var)) -
                self.num_var * scipy.log(self.ub - self.lb)
            )
    
    @property
    def bounds(self):
        return [(self.lb, self.ub)] * self.num_var
    
    def sample_u(self, q):
        r"""Extract a sample from random variates uniform on :math:`[0, 1]`.
        
        For a univariate distribution, this is simply evaluating the inverse
        CDF. To facilitate efficient sampling, this function returns a *vector*
        of PPF values, one value for each variable. Basically, the idea is that,
        given a vector :math:`q` of `num_params` values each of which is
        distributed uniformly on :math:`[0, 1]`, this function will return
        corresponding samples for each variable.
        
        Parameters
        ----------
        q : array of float
            Values between 0 and 1 to evaluate inverse CDF at.
        """
        q = scipy.atleast_1d(q)
        if len(q) != self.num_var:
            raise ValueError("length of q must equal the number of parameters!")
        if q.ndim != 1:
            raise ValueError("q must be one-dimensional!")
        if (q < 0).any() or (q > 1).any():
            raise ValueError("q must be within [0, 1]!")
        
        # Old way, not quite correct:
        # q = scipy.sort(q)
        # return scipy.asarray([(self.ub - self.lb) * v + self.lb for v in q])
        
        # New way, based on conditional marginals:
        out = scipy.zeros_like(q, dtype=float)
        out[0] = self.lb
        for d in xrange(0, len(out)):
            out[d] = (
                (1.0 - (1.0 - q[d])**(1.0 / (self.num_var - d))) *
                (self.ub - out[max(d - 1, 0)]) + out[max(d - 1, 0)]
            )
        return out
    
    def elementwise_cdf(self, p):
        r"""Convert a sample to random variates uniform on :math:`[0, 1]`.
        
        For a univariate distribution, this is simply evaluating the CDF. To
        facilitate efficient sampling, this function returns a *vector* of CDF
        values, one value for each variable. Basically, the idea is that, given
        a vector :math:`q` of `num_params` values each of which is distributed
        according to the prior, this function will return variables uniform on
        :math:`[0, 1]` corresponding to each variable. This is the inverse
        operation to :py:meth:`sample_u`.
        
        Parameters
        ----------
        p : array-like, (`num_params`,)
            Values to evaluate CDF at.
        """
        p = scipy.atleast_1d(p)
        if len(p) != len(self.bounds):
            raise ValueError("length of p must equal the number of parameters!")
        if p.ndim != 1:
            raise ValueError("p must be one-dimensional!")
        c = scipy.zeros(len(self.bounds))
        
        # Old way, based on sorted uniform variables:
        # for k in xrange(0, len(self.bounds)):
        #     if p[k] <= self.bounds[k][0]:
        #         c[k] = 0.0
        #     elif p[k] >= self.bounds[k][1]:
        #         c[k] = 1.0
        #     else:
        #         c[k] = (p[k] - self.bounds[k][0]) / (self.bounds[k][1] - self.bounds[k][0])
        
        # New way, based on conditional marginals:
        for d in xrange(0, len(c)):
            pdm1 = p[d - 1] if d > 0 else self.lb
            if p[d] <= pdm1:
                c[d] = 0.0
            elif p[d] >= self.ub:
                c[d] = 1.0
            else:
                c[d] = 1.0 - (1.0 - (p[d] - pdm1) / (self.ub - pdm1))**(self.num_var - d)
        
        return c
    
    def random_draw(self, size=None):
        """Draw random samples of the hyperparameters.
        
        Parameters
        ----------
        size : None, int or array-like, optional
            The number/shape of samples to draw. If None, only one sample is
            returned. Default is None.
        """
        if size is None:
            size = 1
            single_val = True
        else:
            single_val = False
        
        out_shape = [self.num_var]
        try:
            out_shape.extend(size)
        except TypeError:
            out_shape.append(size)
        
        out = scipy.sort(
            numpy.random.uniform(
                low=self.lb,
                high=self.ub,
                size=out_shape
            ),
            axis=0
        )
        if not single_val:
            return out
        else:
            return out.ravel()

def wrap_fmin_slsqp(fun, guess, opt_kwargs={}):
    """Wrapper for :py:func:`fmin_slsqp` to allow it to be called with :py:func:`minimize`-like syntax.

    This is included to enable the code to run with :py:mod:`scipy` versions
    older than 0.11.0.

    Accepts `opt_kwargs` in the same format as used by
    :py:func:`scipy.optimize.minimize`, with the additional precondition
    that the keyword `method` has already been removed by the calling code.

    Parameters
    ----------
    fun : callable
        The function to minimize.
    guess : sequence
        The initial guess for the parameters.
    opt_kwargs : dict, optional
        Dictionary of extra keywords to pass to
        :py:func:`scipy.optimize.minimize`. Refer to that function's
        docstring for valid options. The keywords 'jac', 'hess' and 'hessp'
        are ignored. Note that if you were planning to use `jac` = True
        (i.e., optimization function returns Jacobian) and have set
        `args` = (True,) to tell :py:meth:`update_hyperparameters` to
        compute and return the Jacobian this may cause unexpected behavior.
        Default is: {}.

    Returns
    -------
    Result : namedtuple
        :py:class:`namedtuple` that mimics the fields of the
        :py:class:`Result` object returned by
        :py:func:`scipy.optimize.minimize`. Has the following fields:

        ======= ======= ===================================================================================
        status  int     Code indicating the exit mode of the optimizer (`imode` from :py:func:`fmin_slsqp`)
        success bool    Boolean indicating whether or not the optimizer thinks a minimum was found.
        fun     float   Value of the optimized function (-1*LL).
        x       ndarray Optimal values of the hyperparameters.
        message str     String describing the exit state (`smode` from :py:func:`fmin_slsqp`)
        nit     int     Number of iterations.
        ======= ======= ===================================================================================

    Raises
    ------
    ValueError
        Invalid constraint type in `constraints`. (See documentation for :py:func:`scipy.optimize.minimize`.)
    """
    opt_kwargs = dict(opt_kwargs)

    opt_kwargs.pop('method', None)

    eqcons = []
    ieqcons = []
    if 'constraints' in opt_kwargs:
        if isinstance(opt_kwargs['constraints'], dict):
            opt_kwargs['constraints'] = [opt_kwargs['constraints'],]
        for con in opt_kwargs.pop('constraints'):
            if con['type'] == 'eq':
                eqcons += [con['fun'],]
            elif con['type'] == 'ineq':
                ieqcons += [con['fun'],]
            else:
                raise ValueError("Invalid constraint type %s!" % (con['type'],))

    if 'jac' in opt_kwargs:
        warnings.warn("Jacobian not supported for default solver SLSQP!",
                      RuntimeWarning)
        opt_kwargs.pop('jac')

    if 'tol' in opt_kwargs:
        opt_kwargs['acc'] = opt_kwargs.pop('tol')

    if 'options' in opt_kwargs:
        opts = opt_kwargs.pop('options')
        opt_kwargs = dict(opt_kwargs.items() + opts.items())

    # Other keywords with less likelihood for causing failures are silently ignored:
    opt_kwargs.pop('hess', None)
    opt_kwargs.pop('hessp', None)
    opt_kwargs.pop('callback', None)

    out, fx, its, imode, smode = scipy.optimize.fmin_slsqp(
        fun,
        guess,
        full_output=True,
        eqcons=eqcons,
        ieqcons=ieqcons,
        **opt_kwargs
    )

    Result = collections.namedtuple('Result',
                                    ['status', 'success', 'fun', 'x', 'message', 'nit'])

    return Result(status=imode,
                  success=(imode == 0),
                  fun=fx,
                  x=out,
                  message=smode,
                  nit=its)

def fixed_poch(a, n):
    """Implementation of the Pochhammer symbol :math:`(a)_n` which handles negative integer arguments properly.
    
    Need conditional statement because scipy's impelementation of the Pochhammer
    symbol is wrong for negative integer arguments. This function uses the
    definition from
    http://functions.wolfram.com/GammaBetaErf/Pochhammer/02/
    
    Parameters
    ----------
    a : float
        The argument.
    n : nonnegative int
        The order.
    """
    # Old form, calls gamma function:
    # if a < 0.0 and a % 1 == 0 and n <= -a:
    #     p = (-1.0)**n * scipy.misc.factorial(-a) / scipy.misc.factorial(-a - n)
    # else:
    #     p = scipy.special.poch(a, n)
    # return p
    if (int(n) != n) or (n < 0):
        raise ValueError("Parameter n must be a nonnegative int!")
    n = int(n)
    # Direct form based on product:
    terms = [a + k for k in range(0, n)]
    return scipy.prod(terms)

def Kn2Der(nu, y, n=0):
    r"""Find the derivatives of :math:`K_\nu(y^{1/2})`.
    
    Parameters
    ----------
    nu : float
        The order of the modified Bessel function of the second kind.
    y : array of float
        The values to evaluate at.
    n : nonnegative int, optional
        The order of derivative to take.
    """
    n = int(n)
    y = scipy.asarray(y, dtype=float)
    sqrty = scipy.sqrt(y)
    if n == 0:
        K = scipy.special.kv(nu, sqrty)
    else:
        K = scipy.zeros_like(y)
        x = scipy.asarray(
            [
                fixed_poch(1.5 - j, j) * y**(0.5 - j)
                for j in scipy.arange(1.0, n + 1.0, dtype=float)
            ]
        ).T
        for k in scipy.arange(1.0, n + 1.0, dtype=float):
            K += (
                scipy.special.kvp(nu, sqrty, n=int(k)) *
                incomplete_bell_poly(n, int(k), x)
            )
    return K

def yn2Kn2Der(nu, y, n=0, tol=5e-4, nterms=1, nu_step=0.001):
    r"""Computes the function :math:`y^{\nu/2} K_{\nu}(y^{1/2})` and its derivatives.
    
    Care has been taken to handle the conditions at :math:`y=0`.
    
    For `n=0`, uses a direct evaluation of the expression, replacing points
    where `y=0` with the appropriate value. For `n>0`, uses a general sum
    expression to evaluate the expression, and handles the value at `y=0` using
    a power series expansion. Where it becomes infinite, the infinities will
    have the appropriate sign for a limit approaching zero from the right.
    
    Uses a power series expansion around :math:`y=0` to avoid numerical issues.
    
    Handles integer `nu` by performing a linear interpolation between values of
    `nu` slightly above and below the requested value.
    
    Parameters
    ----------
    nu : float
        The order of the modified Bessel function and the exponent of `y`.
    y : array of float
        The points to evaluate the function at. These are assumed to be
        nonegative.
    n : nonnegative int, optional
        The order of derivative to take. Set to zero (the default) to get the
        value.
    tol : float, optional
        The distance from zero for which the power series is used. Default is
        5e-4.
    nterms : int, optional
        The number of terms to include in the power series. Default is 1.
    nu_step : float, optional
        The amount to vary `nu` by when handling integer values of `nu`. Default
        is 0.001.
    """
    n = int(n)
    y = scipy.asarray(y, dtype=float)
    
    if n == 0:
        K = y**(nu / 2.0) * scipy.special.kv(nu, scipy.sqrt(y))
        K[y == 0.0] = scipy.special.gamma(nu) / 2.0**(1.0 - nu)
    else:
        K = scipy.zeros_like(y)
        for k in scipy.arange(0.0, n + 1.0, dtype=float):
            K += (
                scipy.special.binom(n, k) * fixed_poch(1.0 + nu / 2.0 - k, k) *
                y**(nu / 2.0 - k) * Kn2Der(nu, y, n=n-k)
            )
        # Do the extra work to handle y == 0 only if we need to:
        mask = (y == 0.0)
        if (mask).any():
            if int(nu) == nu:
                K[mask] = 0.5 * (
                    yn2Kn2Der(nu - nu_step, y[mask], n=n, tol=tol, nterms=nterms, nu_step=nu_step) +
                    yn2Kn2Der(nu + nu_step, y[mask], n=n, tol=tol, nterms=nterms, nu_step=nu_step)
                )
            else:
                if n > nu:
                    K[mask] = scipy.special.gamma(-nu) * fixed_poch(1 + nu - n, n) * scipy.inf
                else:
                    K[mask] = scipy.special.gamma(nu) * scipy.special.gamma(n + 1.0) / (
                        2.0**(1.0 - nu + 2.0 * n) * fixed_poch(1.0 - nu, n) *
                        scipy.special.factorial(n)
                    )
    if tol > 0.0:
        # Replace points within tol (absolute distance) of zero with the power
        # series approximation:
        mask = (y <= tol) & (y > 0.0)
        K[mask] = 0.0
        if int(nu) == nu:
            K[mask] = 0.5 * (
                yn2Kn2Der(nu - nu_step, y[mask], n=n, tol=tol, nterms=nterms, nu_step=nu_step) +
                yn2Kn2Der(nu + nu_step, y[mask], n=n, tol=tol, nterms=nterms, nu_step=nu_step)
            )
        else:
            for k in scipy.arange(n, n + nterms, dtype=float):
                K[mask] += (
                    scipy.special.gamma(nu) * fixed_poch(1.0 + k - n, n) * y[mask]**(k - n) / (
                        2.0**(1.0 - nu + 2 * k) * fixed_poch(1.0 - nu, k) * scipy.special.factorial(k))
                    )
            for k in scipy.arange(0, nterms, dtype=float):
                K[mask] += (
                    scipy.special.gamma(-nu) * fixed_poch(1.0 + nu + k - n, n) *
                    y[mask]**(nu + k - n) / (
                        2.0**(1.0 + nu + 2.0 * k) * fixed_poch(1.0 + nu, k) *
                        scipy.special.factorial(k)
                    )
                )
    
    return K

def incomplete_bell_poly(n, k, x):
    r"""Recursive evaluation of the incomplete Bell polynomial :math:`B_{n, k}(x)`.
    
    Evaluates the incomplete Bell polynomial :math:`B_{n, k}(x_1, x_2, \dots, x_{n-k+1})`,
    also known as the partial Bell polynomial or the Bell polynomial of the
    second kind. This polynomial is useful in the evaluation of (the univariate)
    Faa di Bruno's formula which generalizes the chain rule to higher order
    derivatives.
    
    The implementation here is based on the implementation in:
    :py:func:`sympy.functions.combinatorial.numbers.bell._bell_incomplete_poly`
    Following that function's documentation, the polynomial is computed
    according to the recurrence formula:
    
    .. math::
        
        B_{n, k}(x_1, x_2, \dots, x_{n-k+1}) = \sum_{m=1}^{n-k+1}x_m\binom{n-1}{m-1}B_{n-m, k-1}(x_1, x_2, \dots, x_{n-m-k})
        
    | The end cases are:
    | :math:`B_{0, 0} = 1`
    | :math:`B_{n, 0} = 0` for :math:`n \ge 1`
    | :math:`B_{0, k} = 0` for :math:`k \ge 1`
    
    Parameters
    ----------
    n : scalar int
        The first subscript of the polynomial.
    k : scalar int
        The second subscript of the polynomial.
    x : :py:class:`Array` of floats, (`p`, `n` - `k` + 1)
        `p` sets of `n` - `k` + 1 points to use as the arguments to
        :math:`B_{n,k}`. The second dimension can be longer than
        required, in which case the extra entries are silently ignored
        (this facilitates recursion without needing to subset the array `x`).
    
    Returns
    -------
    result : :py:class:`Array`, (`p`,)
        Incomplete Bell polynomial evaluated at the desired values.
    """
    if n == 0 and k == 0:
        return scipy.ones(x.shape[0], dtype=float)
    elif k == 0 and n >= 1:
        return scipy.zeros(x.shape[0], dtype=float)
    elif n == 0 and k >= 1:
        return scipy.zeros(x.shape[0], dtype=float)
    else:
        result = scipy.zeros(x.shape[0], dtype=float)
        for m in xrange(0, n - k + 1):
            result += x[:, m] * scipy.special.binom(n - 1, m) * incomplete_bell_poly(n - (m + 1), k - 1, x)
        return result

def generate_set_partition_strings(n):
    """Generate the restricted growth strings for all of the partitions of an `n`-member set.
    
    Uses Algorithm H from page 416 of volume 4A of Knuth's `The Art of Computer
    Programming`. Returns the partitions in lexicographical order.
    
    Parameters
    ----------
    n : scalar int, non-negative
        Number of (unique) elements in the set to be partitioned.
    
    Returns
    -------
    partitions : list of :py:class:`Array`
        List has a number of elements equal to the `n`-th Bell number (i.e.,
        the number of partitions for a set of size `n`). Each element has
        length `n`, the elements of which are the restricted growth strings
        describing the partitions of the set. The strings are returned in
        lexicographic order.
    """
    # Handle edge cases:
    if n == 0:
        return []
    elif n == 1:
        return [scipy.array([0])]
    
    partitions = []
    
    # Step 1: Initialize
    a = scipy.zeros(n, dtype=int)
    b = scipy.ones(n, dtype=int)
    
    while True:
        # Step 2: Visit
        partitions.append(a.copy())
        if a[-1] == b[-1]:
            # Step 4: Find j. j is the index of the first element from the end
            # for which a != b, with the exception of the last element.
            j = (a[:-1] != b[:-1]).nonzero()[0][-1]
            # Step 5: Increase a_j (or terminate):
            if j == 0:
                break
            else:
                a[j] += 1
                # Step 6: Zero out a_{j+1} to a_n:
                b[-1] = b[j] + (a[j] == b[j])
                a[j + 1:] = 0
                b[j + 1 :-1] = b[-1]
        else:
            # Step 3: Increase a_n:
            a[-1] += 1
    
    return partitions

def generate_set_partitions(set_):
    """Generate all of the partitions of a set.
    
    This is a helper function that utilizes the restricted growth strings from
    :py:func:`generate_set_partition_strings`. The partitions are returned in
    lexicographic order.
    
    Parameters
    ----------
    set_ : :py:class:`Array` or other Array-like, (`m`,)
        The set to find the partitions of.
    
    Returns
    -------
    partitions : list of lists of :py:class:`Array`
        The number of elements in the outer list is equal to the number of
        partitions, which is the len(`m`)^th Bell number. Each of the inner lists
        corresponds to a single possible partition. The length of an inner list
        is therefore equal to the number of blocks. Each of the arrays in an
        inner list is hence a block.
    """
    set_ = scipy.asarray(set_)
    strings = generate_set_partition_strings(len(set_))
    partitions = []
    for string in strings:
        blocks = []
        for block_num in scipy.unique(string):
            blocks.append(set_[string == block_num])
        partitions.append(blocks)
    
    return partitions

def powerset(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    
    From itertools documentation, https://docs.python.org/2/library/itertools.html.
    """
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))

def unique_rows(arr, return_index=False, return_inverse=False):
    """Returns a copy of arr with duplicate rows removed.
    
    From Stackoverflow "Find unique rows in numpy.array."
    
    Parameters
    ----------
    arr : :py:class:`Array`, (`m`, `n`)
        The array to find the unique rows of.
    return_index : bool, optional
        If True, the indices of the unique rows in the array will also be
        returned. I.e., unique = arr[idx]. Default is False (don't return
        indices).
    return_inverse: bool, optional
        If True, the indices in the unique array to reconstruct the original
        array will also be returned. I.e., arr = unique[inv]. Default is False
        (don't return inverse).
    
    Returns
    -------
    unique : :py:class:`Array`, (`p`, `n`) where `p` <= `m`
        The array `arr` with duplicate rows removed.
    """
    b = scipy.ascontiguousarray(arr).view(
        scipy.dtype((scipy.void, arr.dtype.itemsize * arr.shape[1]))
    )
    try:
        out = scipy.unique(b, return_index=True, return_inverse=return_inverse)
        dum = out[0]
        idx = out[1]
        if return_inverse:
            inv = out[2]
    except TypeError:
        if return_inverse:
            raise RuntimeError(
                "Error in scipy.unique on older versions of numpy prevents "
                "return_inverse from working!"
            )
        # Handle bug in numpy 1.6.2:
        rows = [_Row(row) for row in b]
        srt_idx = sorted(range(len(rows)), key=rows.__getitem__)
        rows = scipy.asarray(rows)[srt_idx]
        row_cmp = [-1]
        for k in xrange(1, len(srt_idx)):
            row_cmp.append(rows[k-1].__cmp__(rows[k]))
        row_cmp = scipy.asarray(row_cmp)
        transition_idxs = scipy.where(row_cmp != 0)[0]
        idx = scipy.asarray(srt_idx)[transition_idxs]
    out = arr[idx]
    if return_index:
        out = (out, idx)
    elif return_inverse:
        out = (out, inv)
    elif return_index and return_inverse:
        out = (out, idx, inv)
    return out

class _Row(object):
    """Helper class to compare rows of a matrix.
    
    This is used to workaround the bug with scipy.unique in numpy 1.6.2.
    
    Parameters
    ----------
    row : ndarray
        The row this object is to represent. Must be 1d. (Will be flattened.)
    """
    def __init__(self, row):
        self.row = scipy.asarray(row).flatten()
    
    def __cmp__(self, other):
        """Compare two rows.
        
        Parameters
        ----------
        other : :py:class:`_Row`
            The row to compare to.
        
        Returns
        -------
        cmp : int
            == ==================================================================
            0  if the two rows have all elements equal
            1  if the first non-equal element (from the right) in self is greater
            -1 if the first non-equal element (from the right) in self is lesser
            == ==================================================================
        """
        if (self.row == other.row).all():
            return 0
        else:
            # Get first non-equal element:
            first_nonequal_idx = scipy.where(self.row != other.row)[0][0]
            if self.row[first_nonequal_idx] > other.row[first_nonequal_idx]:
                return 1
            else:
                # Other must be greater than self in this case:
                return -1

# Conversion factor to get from interquartile range to standard deviation:
IQR_TO_STD = 2.0 * scipy.stats.norm.isf(0.25)

def compute_stats(vals, check_nan=False, robust=False, axis=1, plot_QQ=False, bins=15, name=''):
    """Compute the average statistics (mean, std dev) for the given values.
    
    Parameters
    ----------
    vals : array-like, (`M`, `D`)
        Values to compute the average statistics along the specified axis of.
    check_nan : bool, optional
        Whether or not to check for (and exclude) NaN's. Default is False (do
        not attempt to handle NaN's).
    robust : bool, optional
        Whether or not to use robust estimators (median for mean, IQR for
        standard deviation). Default is False (use non-robust estimators).
    axis : int, optional
        Axis to compute the statistics along. Presently only supported if
        `robust` is False. Default is 1.
    plot_QQ : bool, optional
        Whether or not a QQ plot and histogram should be drawn for each channel.
        Default is False (do not draw QQ plots).
    bins : int, optional
        Number of bins to use when plotting histogram (for plot_QQ=True).
        Default is 15
    name : str, optional
        Name to put in the title of the QQ/histogram plot.
    
    Returns
    -------
    mean : ndarray, (`M`,)
        Estimator for the mean of `vals`.
    std : ndarray, (`M`,)
        Estimator for the standard deviation of `vals`.
    
    Raises
    ------
    NotImplementedError
        If `axis` != 1 when `robust` is True.
    NotImplementedError
        If `plot_QQ` is True.
    """
    if axis != 1 and robust:
        raise NotImplementedError("Values of axis other than 1 are not supported "
                                  "with the robust keyword at this time!")
    if robust:
        # TODO: This stuff should really be vectorized if there is something that allows it!
        if check_nan:
            mean = scipy.stats.nanmedian(vals, axis=axis)
            # TODO: HANDLE AXIS PROPERLY!
            std = scipy.zeros(vals.shape[0], dtype=float)
            for k in xrange(0, len(vals)):
                ch = vals[k]
                ok_idxs = ~scipy.isnan(ch)
                if ok_idxs.any():
                    std[k] = (scipy.stats.scoreatpercentile(ch[ok_idxs], 75) -
                              scipy.stats.scoreatpercentile(ch[ok_idxs], 25))
                else:
                    # Leave a nan where there are no non-nan values:
                    std[k] = scipy.nan
            std /= IQR_TO_STD
        else:
            mean = scipy.median(vals, axis=axis)
            # TODO: HANDLE AXIS PROPERLY!
            std = scipy.asarray([scipy.stats.scoreatpercentile(ch, 75.0) -
                                 scipy.stats.scoreatpercentile(ch, 25.0)
                                 for ch in vals]) / IQR_TO_STD
    else:
        if check_nan:
            mean = scipy.stats.nanmean(vals, axis=axis)
            std = scipy.stats.nanstd(vals, axis=axis)
        else:
            mean = scipy.mean(vals, axis=axis)
            std = scipy.std(vals, axis=axis)
    if plot_QQ:
        f = plt.figure()
        gs = mplgs.GridSpec(2, 2, height_ratios=[8, 1])
        a_QQ = f.add_subplot(gs[0, 0])
        a_hist = f.add_subplot(gs[0, 1])
        a_slider = f.add_subplot(gs[1, :])
        
        title = f.suptitle("")
        
        def update(val):
            """Update the index from the results to be displayed.
            """
            a_QQ.clear()
            a_hist.clear()
            idx = slider.val
            title.set_text("%s, n=%d" % (name, idx))
            
            nan_idxs = scipy.isnan(vals[idx, :])
            if not nan_idxs.all():
                osm, osr = scipy.stats.probplot(vals[idx, ~nan_idxs], dist='norm', plot=None, fit=False)
                a_QQ.plot(osm, osr, 'bo', markersize=10)
                a_QQ.set_title('QQ plot')
                a_QQ.set_xlabel('quantiles of $\mathcal{N}(0,1)$')
                a_QQ.set_ylabel('quantiles of data')
                
                a_hist.hist(vals[idx, ~nan_idxs], bins=bins, normed=True)
                locs = scipy.linspace(vals[idx, ~nan_idxs].min(), vals[idx, ~nan_idxs].max())
                a_hist.plot(locs, scipy.stats.norm.pdf(locs, loc=mean[idx], scale=std[idx]))
                a_hist.set_title('Normalized histogram and reported PDF')
                a_hist.set_xlabel('value')
                a_hist.set_ylabel('density')
            
            f.canvas.draw()
        
        def arrow_respond(slider, event):
            """Event handler for arrow key events in plot windows.

            Pass the slider object to update as a masked argument using a lambda function::

                lambda evt: arrow_respond(my_slider, evt)

            Parameters
            ----------
            slider : Slider instance associated with this handler.
            event : Event to be handled.
            """
            if event.key == 'right':
                slider.set_val(min(slider.val + 1, slider.valmax))
            elif event.key == 'left':
                slider.set_val(max(slider.val - 1, slider.valmin))

        slider = mplw.Slider(a_slider,
                             'index',
                             0,
                             len(vals) - 1,
                             valinit=0,
                             valfmt='%d')
        slider.on_changed(update)
        update(0)
        f.canvas.mpl_connect('key_press_event', lambda evt: arrow_respond(slider, evt))
    
    return (mean, std)

def univariate_envelope_plot(x, mean, std, ax=None, base_alpha=0.375, envelopes=[1, 3], lb=None, ub=None, expansion=10, **kwargs):
    """Make a plot of a mean curve with uncertainty envelopes.
    """
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1)
    elif ax == 'gca':
        ax = plt.gca()
    
    mean = scipy.asarray(mean, dtype=float).copy()
    std = scipy.asarray(std, dtype=float).copy()
    
    # Truncate the data so matplotlib doesn't die:
    if lb is not None and ub is not None and expansion != 1.0:
        expansion *= ub - lb
        ub = ub + expansion
        lb = lb - expansion
    if ub is not None:
        mean[mean > ub] = ub
    if lb is not None:
        mean[mean < lb] = lb
    
    l = ax.plot(x, mean, **kwargs)
    color = plt.getp(l[0], 'color')
    e = []
    for i in envelopes:
        lower = mean - i * std
        upper = mean + i * std
        if ub is not None:
            lower[lower > ub] = ub
            upper[upper > ub] = ub
        if lb is not None:
            lower[lower < lb] = lb
            upper[upper < lb] = lb
        e.append(ax.fill_between(x, lower, upper, facecolor=color, alpha=base_alpha / i))
    return (l, e)

def summarize_sampler(sampler, weights=None, burn=0, ci=0.95, chain_mask=None):
    r"""Create summary statistics of the flattened chain of the sampler.
    
    The confidence regions are computed from the quantiles of the data.
    
    Parameters
    ----------
    sampler : :py:class:`emcee.Sampler` instance or array, (`n_temps`, `n_chains`, `n_samp`, `n_dim`), (`n_chains`, `n_samp`, `n_dim`) or (`n_samp`, `n_dim`)
        The sampler to summarize the chains of.
    weights : array, (`n_temps`, `n_chains`, `n_samp`), (`n_chains`, `n_samp`) or (`n_samp`,), optional
        The weight for each sample. This is useful for post-processing the
        output from MultiNest sampling, for instance.
    burn : int, optional
        The number of samples to burn from the beginning of the chain. Default
        is 0 (no burn).
    ci : float, optional
        A number between 0 and 1 indicating the confidence region to compute.
        Default is 0.95 (return upper and lower bounds of the 95% confidence
        interval).
    chain_mask : (index) array, optional
        Mask identifying the chains to keep before plotting, in case there are
        bad chains. Default is to use all chains.
    
    Returns
    -------
    mean : array, (num_params,)
        Mean values of each of the parameters sampled.
    ci_l : array, (num_params,)
        Lower bounds of the `ci*100%` confidence intervals.
    ci_u : array, (num_params,)
        Upper bounds of the `ci*100%` confidence intervals.
    """
    try:
        k = sampler.flatchain.shape[-1]
    except AttributeError:
        # Assumes array input is only case where there is no "flatchain" attribute.
        k = sampler.shape[-1]
    
    if isinstance(sampler, emcee.EnsembleSampler):
        if chain_mask is None:
            chain_mask = scipy.ones(sampler.chain.shape[0], dtype=bool)
        flat_trace = sampler.chain[chain_mask, burn:, :]
        flat_trace = flat_trace.reshape((-1, k))
    elif isinstance(sampler, emcee.PTSampler):
        if chain_mask is None:
            chain_mask = scipy.ones(sampler.nwalkers, dtype=bool)
        flat_trace = sampler.chain[temp_idx, chain_mask, burn:, :]
        flat_trace = flat_trace.reshape((-1, k))
    elif isinstance(sampler, scipy.ndarray):
        if sampler.ndim == 4:
            if chain_mask is None:
                chain_mask = scipy.ones(sampler.shape[1], dtype=bool)
            flat_trace = sampler[temp_idx, chain_mask, burn:, :]
            flat_trace = flat_trace.reshape((-1, k))
            if weights is not None:
                weights = weights[temp_idx, chain_mask, burn:]
                weights = weights.ravel()
        elif sampler.ndim == 3:
            if chain_mask is None:
                chain_mask = scipy.ones(sampler.shape[0], dtype=bool)
            flat_trace = sampler[chain_mask, burn:, :]
            flat_trace = flat_trace.reshape((-1, k))
            if weights is not None:
                weights = weights[chain_mask, burn:]
                weights = weights.ravel()
        elif sampler.ndim == 2:
            flat_trace = sampler[burn:, :]
            flat_trace = flat_trace.reshape((-1, k))
            if weights is not None:
                weights = weights[burn:]
                weights = weights.ravel()
    else:
        raise ValueError("Unknown sampler class: %s" % (type(sampler),))
    
    cibdry = 100.0 * (1.0 - ci) / 2.0
    if weights is None:
        mean = scipy.mean(flat_trace, axis=0)
        ci_l, ci_u = scipy.percentile(flat_trace, [cibdry, 100.0 - cibdry], axis=0)
    else:
        mean = weights.dot(flat_trace) / weights.sum()
        ci_l = scipy.zeros(k)
        ci_u = scipy.zeros(k)
        p = scipy.asarray([cibdry, 100.0 - cibdry])
        for i in range(0, k):
            srt = flat_trace[:, i].argsort()
            x = flat_trace[srt, i]
            w = weights[srt]
            Sn = w.cumsum()
            pn = 100.0 / Sn[-1] * (Sn - w / 2.0)
            j = scipy.digitize(p, pn) - 1
            ci_l[i], ci_u[i] = x[j] + (p - pn[j]) / (pn[j + 1] - pn[j]) * (x[j + 1] - x[j])
    
    return (mean, ci_l, ci_u)

def plot_sampler(sampler, weights=None, cutoff_weight=None, labels=None, burn=0,
                 chain_mask=None, bins=50, points=None, covs=None, colors=None,
                 ci=[0.95], plot_samples=False, plot_hist=True, chain_alpha=0.1,
                 temp_idx=0, label_fontsize=14, ticklabel_fontsize=9,
                 chain_label_fontsize=9, chain_ticklabel_fontsize=7,
                 suptitle=None, bottom_sep=0.075, label_chain_y=False,
                 max_chain_ticks=6, max_hist_ticks=None, chain_ytick_pad=2.0,
                 suptitle_space=0.1, fixed_width=None, fixed_height=None,
                 ax_space=0.1, cmap='gray_r', hide_chain_ylabels=False,
                 plot_chains=True):
    """Plot the results of MCMC sampler (posterior and chains).
    
    Loosely based on triangle.py. Provides extensive options to format the plot.
    
    Parameters
    ----------
    sampler : :py:class:`emcee.Sampler` instance or array, (`n_temps`, `n_chains`, `n_samp`, `n_dim`), (`n_chains`, `n_samp`, `n_dim`) or (`n_samp`, `n_dim`)
        The sampler to plot the chains/marginals of. Can also be an array of
        samples which matches the shape of the `chain` attribute that would be
        present in a :py:class:`emcee.Sampler` instance.
    weights : array, (`n_temps`, `n_chains`, `n_samp`), (`n_chains`, `n_samp`) or (`n_samp`,), optional
        The weight for each sample. This is useful for post-processing the
        output from MultiNest sampling, for instance.
    cutoff_weight : float, optional
        If `weights` and `cutoff_weight` are present, points with
        `weights < cutoff_weight * weights.max()` will be excluded. Default is
        to plot all points.
    labels : list of str, optional
        The labels to use for each of the free parameters. Default is to leave
        the axes unlabeled.
    burn : int, optional
        The number of samples to burn before making the marginal histograms.
        Default is zero (use all samples).
    chain_mask : (index) array, optional
        Mask identifying the chains to keep before plotting, in case there are
        bad chains. Default is to use all chains.
    points : array, (`D`,) or (`N`, `D`), optional
        Array of point(s) to plot onto each marginal and chain. Default is None.
    covs : array, (`D`, `D`) or (`N`, `D`, `D`), optional
        Covariance matrix or array of covariance matrices to plot onto each
        marginal. If you do not want to plot a covariance matrix for a specific
        point, set its corresponding entry to `None`. Default is to not plot
        confidence ellipses for any points.
    ci : array, (`num_ci`,), optional
        List of confidence intervals to plot for each non-`None` entry in `covs`.
        Default is 0.95 (just plot the 95 percent confidence interval).
    plot_samples : bool, optional
        If True, the samples are plotted as individual points. Default is False.
    chain_alpha : float, optional
        The transparency to use for the plots of the individual chains. Setting
        this to something low lets you better visualize what is going on.
        Default is 0.1.
    temp_idx : int, optional
        Index of the temperature to plot when plotting a
        :py:class:`emcee.PTSampler`. Default is 0 (samples from the posterior).
    label_fontsize : float, optional
        The font size (in points) to use for the axis labels. Default is 16.
    ticklabel_fontsize : float, optional
        The font size (in points) to use for the axis tick labels. Default is 10.
    chain_label_fontsize : float, optional
        The font size (in points) to use for the labels of the chain axes.
        Default is 10.
    chain_ticklabel_fontsize : float, optional
        The font size (in points) to use for the chain axis tick labels. Default
        is 6.
    suptitle : str, optional
        The figure title to place at the top. Default is no title.
    bottom_sep : float, optional
        The separation (in relative figure units) between the chains and the
        marginals. Default is 0.075.
    label_chain_y : bool, optional
        If True, the chain plots will have y axis labels. Default is False.
    max_chain_ticks : int, optional
        The maximum number of y-axis ticks for the chain plots. Default is 6.
    max_hist_ticks : int, optional
        The maximum number of ticks for the histogram plots. Default is None
        (no limit).
    chain_ytick_pad : float, optional
        The padding (in points) between the y-axis tick labels and the axis for
        the chain plots. Default is 2.0.
    suptitle_space : float, optional
        The amount of space (in relative figure units) to leave for a figure
        title. Default is 0.1.
    fixed_width : float, optional
        The desired figure width (in inches). Conflicts with `fixed_height`.
    fixed_height : float, optional
        The desired figure height (in inches). Conflicts with `fixed_width`.
    ax_space : float, optional
        The `w_space` and `h_space` to use. Default is 0.1.
    plot_chains : bool, optional
        If True, plot the sampler chains. Default is True.
    """
    masked_weights = None
    if points is not None:
        points = scipy.atleast_2d(points)
        if covs is not None and len(covs) != len(points):
            raise ValueError(
                "If covariance matrices are provided, len(covs) must equal len(points)!"
            )
        elif covs is None:
            covs = [None,] * len(points)
        if colors is None:
            c_cycle = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
            colors = [c_cycle.next() for p in points]
    # Create axes:
    try:
        k = sampler.flatchain.shape[-1]
    except AttributeError:
        # Assumes array input is only case where there is no "flatchain" attribute.
        k = sampler.shape[-1]
    
    if labels is None:
        labels = [''] * k
    
    if plot_chains:
        fw = 2.0 * (1.0 - suptitle_space - 0.2 - bottom_sep - ax_space) / (0.9 - 2.0 * ax_space) * k
        fh = 2.0 * k
    else:
        fw = 2.0 * (1.0 - suptitle_space - bottom_sep) * k
        fh = 2.0 * k
    
    if fixed_width is not None and fixed_height is not None:
        raise ValueError("Can only pass one of fixed_width and fixed_height!")
    if fixed_width is not None:
        fh = fh / fw * fixed_width
        fw = fixed_width
    elif fixed_height is not None:
        fw = fw / fh * fixed_height
        fh = fixed_height
    f = plt.figure(figsize=(fw, fh))
    if plot_chains:
        gs1 = mplgs.GridSpec(k, k)
        gs2 = mplgs.GridSpec(1, k)
        gs1.update(bottom=0.2 + bottom_sep, top=1.0 - suptitle_space, left=0.1, right=0.9, wspace=ax_space, hspace=ax_space)
        gs2.update(bottom=0.1, top=0.2, left=0.1, right=0.9, wspace=ax_space, hspace=ax_space)
    else:
        gs1 = mplgs.GridSpec(k, k)
        gs1.update(bottom=bottom_sep, top=1.0 - suptitle_space, left=0.1, right=0.9, wspace=ax_space, hspace=ax_space)
    axes = []
    # j is the row, i is the column.
    for j in xrange(0, k + int(plot_chains)):
        row = []
        for i in xrange(0, k):
            if i > j:
                row.append(None)
            else:
                sharey = row[-1] if i > 0 and i < j and j < k else None
                sharex = axes[-1][i] if j > i and j < k else \
                    (row[-1] if i > 0 and j == k else None)
                gs = gs1[j, i] if j < k else gs2[:, i]
                row.append(f.add_subplot(gs, sharey=sharey, sharex=sharex))
                row[-1].tick_params(labelsize=ticklabel_fontsize if j < k else chain_ticklabel_fontsize)
        axes.append(row)
    axes = scipy.asarray(axes)
    
    # Update axes with the data:
    if isinstance(sampler, emcee.EnsembleSampler):
        if chain_mask is None:
            chain_mask = scipy.ones(sampler.chain.shape[0], dtype=bool)
        flat_trace = sampler.chain[chain_mask, burn:, :]
        flat_trace = flat_trace.reshape((-1, k))
    elif isinstance(sampler, emcee.PTSampler):
        if chain_mask is None:
            chain_mask = scipy.ones(sampler.nwalkers, dtype=bool)
        flat_trace = sampler.chain[temp_idx, chain_mask, burn:, :]
        flat_trace = flat_trace.reshape((-1, k))
    elif isinstance(sampler, scipy.ndarray):
        if sampler.ndim == 4:
            if chain_mask is None:
                chain_mask = scipy.ones(sampler.shape[1], dtype=bool)
            flat_trace = sampler[temp_idx, chain_mask, burn:, :]
            flat_trace = flat_trace.reshape((-1, k))
            if weights is not None:
                weights = weights[temp_idx, chain_mask, burn:]
                weights = weights.ravel()
        elif sampler.ndim == 3:
            if chain_mask is None:
                chain_mask = scipy.ones(sampler.shape[0], dtype=bool)
            flat_trace = sampler[chain_mask, burn:, :]
            flat_trace = flat_trace.reshape((-1, k))
            if weights is not None:
                weights = weights[chain_mask, burn:]
                weights = weights.ravel()
        elif sampler.ndim == 2:
            flat_trace = sampler[burn:, :]
            flat_trace = flat_trace.reshape((-1, k))
            if weights is not None:
                weights = weights[burn:]
                weights = weights.ravel()
        if cutoff_weight is not None and weights is not None:
            mask = weights >= cutoff_weight * weights.max()
            flat_trace = flat_trace[mask, :]
            masked_weights = weights[mask]
        else:
            masked_weights = weights
    else:
        raise ValueError("Unknown sampler class: %s" % (type(sampler),))
    
    # j is the row, i is the column.
    for i in xrange(0, k):
        axes[i, i].clear()
        if plot_hist:
            axes[i, i].hist(flat_trace[:, i], bins=bins, color='black', weights=masked_weights, normed=True)
        if plot_samples:
            axes[i, i].plot(flat_trace[:, i], scipy.zeros_like(flat_trace[:, i]), ',', alpha=0.1)
        if points is not None:
            # axvline can only take a scalar x, so we have to loop:
            for p, c, cov in zip(points, colors, covs):
                axes[i, i].axvline(x=p[i], linewidth=3, color=c)
                if cov is not None:
                    xlim = axes[i, i].get_xlim()
                    i_grid = scipy.linspace(xlim[0], xlim[1], 100)
                    axes[i, i].plot(
                        i_grid,
                        scipy.stats.norm.pdf(
                            i_grid,
                            loc=p[i],
                            scale=scipy.sqrt(cov[i, i])
                        ),
                        c,
                        linewidth=3.0
                    )
                    axes[i, i].set_xlim(xlim)
        if i == k - 1:
            axes[i, i].set_xlabel(labels[i], fontsize=label_fontsize)
            plt.setp(axes[i, i].xaxis.get_majorticklabels(), rotation=90)
        if i < k - 1:
            plt.setp(axes[i, i].get_xticklabels(), visible=False)
        plt.setp(axes[i, i].get_yticklabels(), visible=False)
        for j in xrange(i + 1, k):
            axes[j, i].clear()
            if plot_hist:
                ct, x, y, im = axes[j, i].hist2d(
                    flat_trace[:, i],
                    flat_trace[:, j],
                    bins=bins,
                    cmap=cmap,
                    weights=masked_weights
                )
            if plot_samples:
                axes[j, i].plot(flat_trace[:, i], flat_trace[:, j], ',', alpha=0.1)
            if points is not None:
                for p, c, cov in zip(points, colors, covs):
                    axes[j, i].plot(p[i], p[j], 'o', color=c)
                    if cov is not None:
                        Sigma = scipy.asarray([[cov[i, i], cov[i, j]], [cov[j, i], cov[j, j]]], dtype=float)
                        lam, v = scipy.linalg.eigh(Sigma)
                        chi2 = [-scipy.log(1.0 - cival) * 2.0 for cival in ci]
                        a = [2.0 * scipy.sqrt(chi2val * lam[-1]) for chi2val in chi2]
                        b = [2.0 * scipy.sqrt(chi2val * lam[-2]) for chi2val in chi2]
                        ang = scipy.arctan2(v[1, -1], v[0, -1])
                        for aval, bval in zip(a, b):
                            ell = mplp.Ellipse(
                                [p[i], p[j]],
                                aval,
                                bval,
                                angle=scipy.degrees(ang),
                                facecolor='none',
                                edgecolor=c,
                                linewidth=3
                            )
                            axes[j, i].add_artist(ell)
                # axes[j, i].plot(points[i], points[j], 'o')
            # xmid = 0.5 * (x[1:] + x[:-1])
            # ymid = 0.5 * (y[1:] + y[:-1])
            # axes[j, i].contour(xmid, ymid, ct.T, colors='k')
            if j < k - 1:
                plt.setp(axes[j, i].get_xticklabels(), visible=False)
            if i != 0:
                plt.setp(axes[j, i].get_yticklabels(), visible=False)
            if i == 0:
                axes[j, i].set_ylabel(labels[j], fontsize=label_fontsize)
            if j == k - 1:
                axes[j, i].set_xlabel(labels[i], fontsize=label_fontsize)
                plt.setp(axes[j, i].xaxis.get_majorticklabels(), rotation=90)
        if plot_chains:
            axes[-1, i].clear()
            if isinstance(sampler, emcee.EnsembleSampler):
                axes[-1, i].plot(sampler.chain[:, :, i].T, alpha=chain_alpha)
            elif isinstance(sampler, emcee.PTSampler):
                axes[-1, i].plot(sampler.chain[temp_idx, :, :, i].T, alpha=chain_alpha)
            else:
                if sampler.ndim == 4:
                    axes[-1, i].plot(sampler[temp_idx, :, :, i].T, alpha=chain_alpha)
                elif sampler.ndim == 3:
                    axes[-1, i].plot(sampler[:, :, i].T, alpha=chain_alpha)
                elif sampler.ndim == 2:
                    axes[-1, i].plot(sampler[:, i].T, alpha=chain_alpha)
            # Plot the weights on top of the chains:
            if weights is not None:
                a_wt = axes[-1, i].twinx()
                a_wt.plot(weights, alpha=chain_alpha, linestyle='--', color='r')
                plt.setp(a_wt.yaxis.get_majorticklabels(), visible=False)
                a_wt.yaxis.set_ticks_position('none')
                # Plot the cutoff weight as a horizontal line and the first sample
                # which is included as a vertical bar. Note that this won't be quite
                # the right behavior if the weights are not roughly monotonic.
                if cutoff_weight is not None:
                    a_wt.axhline(cutoff_weight * weights.max(), linestyle='-', color='r')
                    wi, = scipy.where(weights >= cutoff_weight * weights.max())
                    a_wt.axvline(wi[0], linestyle='-', color='r')
            if burn > 0:
                axes[-1, i].axvline(burn, color='r', linewidth=3)
            if points is not None:
                for p, c in zip(points, colors):
                    axes[-1, i].axhline(y=p[i], linewidth=3, color=c)
                # Reset the xlim since it seems to get messed up:
                axes[-1, i].set_xlim(left=0)
                # try:
                #     [axes[-1, i].axhline(y=pt, linewidth=3) for pt in points[i]]
                # except TypeError:
                #     axes[-1, i].axhline(y=points[i], linewidth=3)
            if label_chain_y:
                axes[-1, i].set_ylabel(labels[i], fontsize=chain_label_fontsize)
            axes[-1, i].set_xlabel('step', fontsize=chain_label_fontsize)
            plt.setp(axes[-1, i].xaxis.get_majorticklabels(), rotation=90)
            for tick in axes[-1, i].get_yaxis().get_major_ticks():
                tick.set_pad(chain_ytick_pad)
                tick.label1 = tick._get_text1()
    
    for i in xrange(0, k):
        if max_hist_ticks is not None:
            axes[k - 1, i].xaxis.set_major_locator(plt.MaxNLocator(nbins=max_hist_ticks - 1))
            axes[i, 0].yaxis.set_major_locator(plt.MaxNLocator(nbins=max_hist_ticks - 1))
        if plot_chains and max_chain_ticks is not None:
            axes[k, i].yaxis.set_major_locator(plt.MaxNLocator(nbins=max_chain_ticks - 1))
            axes[k, i].xaxis.set_major_locator(plt.MaxNLocator(nbins=max_chain_ticks - 1))
        if plot_chains and hide_chain_ylabels:
            plt.setp(axes[k, i].get_yticklabels(), visible=False)
    
    if suptitle is not None:
        f.suptitle(suptitle)
    f.canvas.draw()
    return f

def plot_sampler_fingerprint(
        sampler, hyperprior, weights=None, cutoff_weight=None, nbins=None,
        labels=None, burn=0, chain_mask=None, temp_idx=0, points=None,
        plot_samples=False, sample_color='k', point_color=None, point_lw=3,
        title='', rot_x_labels=False, figsize=None
    ):
    """Make a plot of the sampler's "fingerprint": univariate marginal histograms for all hyperparameters.
    
    The hyperparameters are mapped to [0, 1] using
    :py:meth:`hyperprior.elementwise_cdf`, so this can only be used with prior
    distributions which implement this function.
    
    Returns the figure and axis created.
    
    Parameters
    ----------
    sampler : :py:class:`emcee.Sampler` instance or array, (`n_temps`, `n_chains`, `n_samp`, `n_dim`), (`n_chains`, `n_samp`, `n_dim`) or (`n_samp`, `n_dim`)
        The sampler to plot the chains/marginals of. Can also be an array of
        samples which matches the shape of the `chain` attribute that would be
        present in a :py:class:`emcee.Sampler` instance.
    hyperprior : :py:class:`~gptools.utils.JointPrior` instance
        The joint prior distribution for the hyperparameters. Used to map the
        values to [0, 1] so that the hyperparameters can all be shown on the
        same axis.
    weights : array, (`n_temps`, `n_chains`, `n_samp`), (`n_chains`, `n_samp`) or (`n_samp`,), optional
        The weight for each sample. This is useful for post-processing the
        output from MultiNest sampling, for instance.
    cutoff_weight : float, optional
        If `weights` and `cutoff_weight` are present, points with
        `weights < cutoff_weight * weights.max()` will be excluded. Default is
        to plot all points.
    nbins : int or array of int, (`D`,), optional
        The number of bins dividing [0, 1] to use for each histogram. If a
        single int is given, this is used for all of the hyperparameters. If an
        array of ints is given, these are the numbers of bins for each of the
        hyperparameters. The default is to determine the number of bins using
        the Freedman-Diaconis rule.
    labels : array of str, (`D`,), optional
        The labels for each hyperparameter. Default is to use empty strings.
    burn : int, optional
        The number of samples to burn before making the marginal histograms.
        Default is zero (use all samples).
    chain_mask : (index) array, optional
        Mask identifying the chains to keep before plotting, in case there are
        bad chains. Default is to use all chains.
    temp_idx : int, optional
        Index of the temperature to plot when plotting a
        :py:class:`emcee.PTSampler`. Default is 0 (samples from the posterior).
    points : array, (`D`,) or (`N`, `D`), optional
        Array of point(s) to plot as horizontal lines. Default is None.
    plot_samples : bool, optional
        If True, the samples are plotted as horizontal lines. Default is False.
    sample_color : str, optional
        The color to plot the samples in. Default is 'k', meaning black.
    point_color : str or list of str, optional
        The color to plot the individual points in. Default is to loop through
        matplotlib's default color sequence. If a list is provided, it will be
        cycled through.
    point_lw : float, optional
        Line width to use when plotting the individual points.
    title : str, optional
        Title to use for the plot.
    rot_x_labels : bool, optional
        If True, the labels for the x-axis are rotated 90 degrees. Default is
        False (do not rotate labels).
    figsize : 2-tuple, optional
        The figure size to use. Default is to use the matplotlib default.
    """
    try:
        k = sampler.flatchain.shape[-1]
    except AttributeError:
        # Assumes array input is only case where there is no "flatchain" attribute.
        k = sampler.shape[-1]
    # Process the samples:
    if isinstance(sampler, emcee.EnsembleSampler):
        if chain_mask is None:
            chain_mask = scipy.ones(sampler.chain.shape[0], dtype=bool)
        flat_trace = sampler.chain[chain_mask, burn:, :]
        flat_trace = flat_trace.reshape((-1, k))
    elif isinstance(sampler, emcee.PTSampler):
        if chain_mask is None:
            chain_mask = scipy.ones(sampler.nwalkers, dtype=bool)
        flat_trace = sampler.chain[temp_idx, chain_mask, burn:, :]
        flat_trace = flat_trace.reshape((-1, k))
    elif isinstance(sampler, scipy.ndarray):
        if sampler.ndim == 4:
            if chain_mask is None:
                chain_mask = scipy.ones(sampler.shape[1], dtype=bool)
            flat_trace = sampler[temp_idx, chain_mask, burn:, :]
            flat_trace = flat_trace.reshape((-1, k))
            if weights is not None:
                weights = weights[temp_idx, chain_mask, burn:]
                weights = weights.ravel()
        elif sampler.ndim == 3:
            if chain_mask is None:
                chain_mask = scipy.ones(sampler.shape[0], dtype=bool)
            flat_trace = sampler[chain_mask, burn:, :]
            flat_trace = flat_trace.reshape((-1, k))
            if weights is not None:
                weights = weights[chain_mask, burn:]
                weights = weights.ravel()
        elif sampler.ndim == 2:
            flat_trace = sampler[burn:, :]
            flat_trace = flat_trace.reshape((-1, k))
            if weights is not None:
                weights = weights[burn:]
                weights = weights.ravel()
        if cutoff_weight is not None and weights is not None:
            mask = weights >= cutoff_weight * weights.max()
            flat_trace = flat_trace[mask, :]
            weights = weights[mask]
    else:
        raise ValueError("Unknown sampler class: %s" % (type(sampler),))
    
    if labels is None:
        labels = [''] * k
    
    u = scipy.asarray([hyperprior.elementwise_cdf(p) for p in flat_trace], dtype=float).T
    if nbins is None:
        lq, uq = scipy.stats.scoreatpercentile(u, [25, 75], axis=1)
        h = 2.0 * (uq - lq) / u.shape[0]**(1.0 / 3.0)
        n = scipy.asarray(scipy.ceil(1.0 / h), dtype=int)
    else:
        try:
            iter(nbins)
            n = nbins
        except TypeError:
            n = nbins * scipy.ones(u.shape[0])
    
    hist = [scipy.stats.histogram(uv, numbins=nv, defaultlimits=[0, 1], weights=weights) for uv, nv in zip(u, n)]
    max_ct = max([max(h.count) for h in hist])
    min_ct = min([min(h.count) for h in hist])
    
    f = plt.figure(figsize=figsize)
    a = f.add_subplot(1, 1, 1)
    for i, (h, pn) in enumerate(zip(hist, labels)):
        a.imshow(
            scipy.atleast_2d(scipy.asarray(h.count[::-1], dtype=float)).T,
            cmap='gray_r',
            interpolation='nearest',
            vmin=min_ct,
            vmax=max_ct,
            extent=(i, i + 1, 0, 1),
            aspect='auto'
        )
    
    if plot_samples:
        for p in u:
            for i, uv in enumerate(p):
                a.plot([i, i + 1], [uv, uv], sample_color, alpha=0.1)
    
    if points is not None:
        points = scipy.atleast_2d(scipy.asarray(points, dtype=float))
        u_points = [hyperprior.elementwise_cdf(p) for p in points]
        if point_color is None:
            c_cycle = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
        else:
            c_cycle = itertools.cycle(scipy.atleast_1d(point_color))
        for p in u_points:
            c = c_cycle.next()
            for i, uv in enumerate(p):
                a.plot([i, i + 1], [uv, uv], color=c, lw=point_lw)
    
    a.set_xlim(0, len(hist))
    a.set_ylim(0, 1)
    a.set_xticks(0.5 + scipy.arange(0, len(hist), dtype=float))
    a.set_xticklabels(labels)
    if rot_x_labels:
        plt.setp(a.xaxis.get_majorticklabels(), rotation=90)
    a.set_xlabel("parameter")
    a.set_ylabel("$u=F_P(p)$")
    a.set_title(title)
    
    return f, a

def plot_sampler_cov(
        sampler, method='corr', weights=None, cutoff_weight=None, labels=None,
        burn=0, chain_mask=None, temp_idx=0, cbar_label=None, title='',
        rot_x_labels=False, figsize=None, xlabel_on_top=True
    ):
    """Make a plot of the sampler's correlation or covariance matrix.
    
    Returns the figure and axis created.
    
    Parameters
    ----------
    sampler : :py:class:`emcee.Sampler` instance or array, (`n_temps`, `n_chains`, `n_samp`, `n_dim`), (`n_chains`, `n_samp`, `n_dim`) or (`n_samp`, `n_dim`)
        The sampler to plot the chains/marginals of. Can also be an array of
        samples which matches the shape of the `chain` attribute that would be
        present in a :py:class:`emcee.Sampler` instance.
    method : {'corr', 'cov'}
        Whether to plot the correlation matrix ('corr') or the covariance matrix
        ('cov'). The covariance matrix is often not useful because different
        parameters have wildly different scales. Default is to plot the
        correlation matrix.
    labels : array of str, (`D`,), optional
        The labels for each hyperparameter. Default is to use empty strings.
    burn : int, optional
        The number of samples to burn before making the marginal histograms.
        Default is zero (use all samples).
    chain_mask : (index) array, optional
        Mask identifying the chains to keep before plotting, in case there are
        bad chains. Default is to use all chains.
    temp_idx : int, optional
        Index of the temperature to plot when plotting a
        :py:class:`emcee.PTSampler`. Default is 0 (samples from the posterior).
    cbar_label : str, optional
        The label to use for the colorbar. The default is chosen based on the
        value of the `method` keyword.
    title : str, optional
        Title to use for the plot.
    rot_x_labels : bool, optional
        If True, the labels for the x-axis are rotated 90 degrees. Default is
        False (do not rotate labels).
    figsize : 2-tuple, optional
        The figure size to use. Default is to use the matplotlib default.
    xlabel_on_top : bool, optional
        If True, the x-axis labels are put on top (the way mathematicians
        present matrices). Default is True.
    """
    try:
        k = sampler.flatchain.shape[-1]
    except AttributeError:
        # Assumes array input is only case where there is no "flatchain" attribute.
        k = sampler.shape[-1]
    # Process the samples:
    if isinstance(sampler, emcee.EnsembleSampler):
        if chain_mask is None:
            chain_mask = scipy.ones(sampler.chain.shape[0], dtype=bool)
        flat_trace = sampler.chain[chain_mask, burn:, :]
        flat_trace = flat_trace.reshape((-1, k))
    elif isinstance(sampler, emcee.PTSampler):
        if chain_mask is None:
            chain_mask = scipy.ones(sampler.nwalkers, dtype=bool)
        flat_trace = sampler.chain[temp_idx, chain_mask, burn:, :]
        flat_trace = flat_trace.reshape((-1, k))
    elif isinstance(sampler, scipy.ndarray):
        if sampler.ndim == 4:
            if chain_mask is None:
                chain_mask = scipy.ones(sampler.shape[1], dtype=bool)
            flat_trace = sampler[temp_idx, chain_mask, burn:, :]
            flat_trace = flat_trace.reshape((-1, k))
            if weights is not None:
                weights = weights[temp_idx, chain_mask, burn:]
                weights = weights.ravel()
        elif sampler.ndim == 3:
            if chain_mask is None:
                chain_mask = scipy.ones(sampler.shape[0], dtype=bool)
            flat_trace = sampler[chain_mask, burn:, :]
            flat_trace = flat_trace.reshape((-1, k))
            if weights is not None:
                weights = weights[chain_mask, burn:]
                weights = weights.ravel()
        elif sampler.ndim == 2:
            flat_trace = sampler[burn:, :]
            flat_trace = flat_trace.reshape((-1, k))
            if weights is not None:
                weights = weights[burn:]
                weights = weights.ravel()
        if cutoff_weight is not None and weights is not None:
            mask = weights >= cutoff_weight * weights.max()
            flat_trace = flat_trace[mask, :]
            weights = weights[mask]
    else:
        raise ValueError("Unknown sampler class: %s" % (type(sampler),))
    
    if labels is None:
        labels = [''] * k
    
    if cbar_label is None:
        cbar_label = r'$\mathrm{cov}(p_1, p_2)$' if method == 'cov' else r'$\mathrm{corr}(p_1, p_2)$'
    
    if weights is None:
        if method == 'corr':
            cov = scipy.corrcoef(flat_trace, rowvar=0, ddof=1)
        else:
            cov = scipy.cov(flat_trace, rowvar=0, ddof=1)
    else:
        cov = scipy.cov(flat_trace, rowvar=0, aweights=weights)
        if method == 'corr':
            stds = scipy.sqrt(scipy.diag(cov))
            STD_1, STD_2 = scipy.meshgrid(stds, stds)
            cov = cov / (STD_1 * STD_2)
    
    f_cov = plt.figure(figsize=figsize)
    a_cov = f_cov.add_subplot(1, 1, 1)
    a_cov.set_title(title)
    if method == 'cov':
        vmax = scipy.absolute(cov).max()
    else:
        vmax = 1.0
    cax = a_cov.pcolor(cov, cmap='seismic', vmin=-1 * vmax, vmax=vmax)
    divider = make_axes_locatable(a_cov)
    a_cb = divider.append_axes("right", size="10%", pad=0.05)
    cbar = f_cov.colorbar(cax, cax=a_cb, label=cbar_label)
    a_cov.set_xlabel('parameter')
    a_cov.set_ylabel('parameter')
    a_cov.axis('square')
    a_cov.invert_yaxis()
    if xlabel_on_top:
        a_cov.xaxis.tick_top()
        a_cov.xaxis.set_label_position('top')
    a_cov.set_xticks(0.5 + scipy.arange(0, flat_trace.shape[1], dtype=float))
    a_cov.set_yticks(0.5 + scipy.arange(0, flat_trace.shape[1], dtype=float))
    a_cov.set_xticklabels(labels)
    if rot_x_labels:
        plt.setp(a_cov.xaxis.get_majorticklabels(), rotation=90)
    a_cov.set_yticklabels(labels)
    a_cov.set_xlim(0, flat_trace.shape[1])
    a_cov.set_ylim(flat_trace.shape[1], 0)
    
    return f_cov, a_cov
