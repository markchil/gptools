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
    import matplotlib.pyplot as plt
    import matplotlib.widgets as mplw
    import matplotlib.gridspec as mplgs
except ImportError:
    warnings.warn("Could not import matplotlib. plot_QQ keyword for compute_stats will not function.",
                  ImportWarning)


class LessThanUniformPotential(object):
    """Class to implement a potential to enforce an inequality constraint.
    
    Specifically lets you change the param with l_idx to have a uniform prior
    between its lower bound and the param with g_idx.
    
    Returns log((ub-lb)/(theta[g_idx]-lb)) if theta[l_idx] <= theta[g_idx],
    double_min otherwise.
    
    Parameters
    ----------
    l_idx : int
        Index of the parameter that is required to be lesser.
    g_idx : int
        Index of the parameter that is required to be greater.
    """
    def __init__(self, l_idx, g_idx):
        self.l_idx = l_idx
        self.g_idx = g_idx
    
    def __call__(self, theta, k):
        """Return the log-density of the potential.
        
        Parameters
        ----------
        theta : array-like
            Array of the hyperparameters.
        k : Kernel instance
            The kernel the hyperparameters apply to.
        
        Returns
        -------
        f : float
            Returns log((ub-lb)/(theta[g_idx]-lb)) if the condition is met, -inf if not.
        """
        if theta[self.l_idx] <= theta[self.g_idx] and theta[self.l_idx] >= k.param_bounds[self.l_idx][0]:
            return (scipy.log(k.param_bounds[self.l_idx][1] - k.param_bounds[self.l_idx][0]) -
                    scipy.log(theta[self.g_idx] - k.param_bounds[self.l_idx][0]))
        else:
            return -scipy.inf

class JeffreysPrior(object):
    """Class to implement a Jeffreys prior over a finite range. Returns log-density.
    
    Parameters
    ----------
    idx : int
        The index this prior applies to.
    bounds : 2-tuple
        The bounds for the parameter this prior corresponds to: (lb, ub).
    """
    def __init__(self, idx, bounds):
        self.idx = idx
        self.bounds = bounds
    
    def __call__(self, theta):
        if self.bounds[0] <= theta[self.idx] and theta[self.idx] <= self.bounds[1]:
            return -scipy.log(scipy.log(self.bounds[1] / self.bounds[0])) - scipy.log(theta[self.idx])
        else:
            return -scipy.inf
            
    def interval(self, alpha):
        if alpha == 1:
            return self.bounds
        else:
            raise ValueError("Unsupported interval!")

class LinearPrior(object):
    """Class to implement a linear prior. Returns log-density.
    
    Parameters
    ----------
    idx : int
        The index this prior applies to.
    bounds : 2-tuple
        The bounds for the parameter this prior corresponds to: (lb, ub).
    """
    def __init__(self, idx, bounds):
        self.bounds = bounds
        self.idx = idx
    
    def __call__(self, theta):
        """Return the log-density of the uniform prior.
        
        Parameters
        ----------
        theta : array-like, or float
            Value of values of the hyperparameter.
        
        Returns
        -------
        f : :py:class:`Array` or float
            Returns log(2/(b-a)^2) + log(b-theta) if theta is in bounds, -inf
            if theta is out of bounds.
        """
        if self.bounds[0] <= theta[self.idx] and theta[self.idx] <= self.bounds[1]:
            return scipy.log(2 / (self.bounds[1] - self.bounds[0])**2) + scipy.log(self.bounds[1] - theta[self.idx])
        else:
            return -scipy.inf
    
    def interval(self, alpha):
        if alpha == 1:
            return self.bounds
        else:
            raise ValueError("Unsupported interval!")

class UniformPrior(object):
    """Class to implement a uniform prior. Returns log-density.
    
    Parameters
    ----------
    idx : int
        The index this prior applies to.
    bounds : 2-tuple
        The bounds for the parameter this prior corresponds to: (lb, ub).
    """
    def __init__(self, idx, bounds):
        self.bounds = bounds
        self.idx = idx
        
    def __call__(self, theta):
        """Return the log-PDF of the uniform prior.
        
        Parameters
        ----------
        theta : array-like
            Values of the hyperparameters.
        
        Returns
        -------
        f : :py:class:`Array` or float
            Returns -log(ub - lb) if theta is scalar and in bounds, double_min
            if theta is scalar and out of bounds and an appropriately-shaped
            array if theta is array-like.
        """
        if self.bounds[0] <= theta[self.idx] and theta[self.idx] <= self.bounds[1]:
            return -scipy.log(self.bounds[1] - self.bounds[0])
        else:
            return -scipy.inf  #scipy.finfo('d').min
    
    def interval(self, alpha):
        # Can't store the frozen distribution since it isn't pickleable.
        return scipy.stats.uniform.interval(alpha, loc=self.bounds[0], scale=self.bounds[1] - self.bounds[0])
    
    def rvs(self, size=None):
        return scipy.stats.uniform.rvs(size=size, loc=self.bounds[0], scale=self.bounds[1] - self.bounds[0])

class JointPrior(object):
    """Abstract class for objects implementing joint priors over hyperparameters.
    """
    
    def __call__(self, theta):
        """Evaluate the prior log-PDF at the given values of the hyperparameters, theta.
        
        Parameters
        ----------
        theta : array-like, (`num_params`,)
            The hyperparameters to evaluate the log-PDF at.
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
    
    def __mul__(self, other):
        """Multiply two :py:class:`JointPrior` instances together.
        """
        return ProductJointPrior(self, other)

class CombinedBounds(object):
    """Object to support reassignment of the bounds from a combined prior.
    """
    # TODO: This could use a lot more work!
    def __init__(self, l1, l2):
        self.l1 = l1
        self.l2 = l2
    
    def __getitem__(self, pos):
        return (list(self.l1) + list(self.l2))[pos]
    
    def __setitem__(self, pos, value):
        if pos < len(self.l1):
            self.l1[pos] = value
        else:
            self.l2[pos - len(self.l1)] = value
    
    def __len__(self):
        return len(self.l1) + len(self.l2)
    
    def __invert__(self):
        return ~scipy.asarray(self)

class MaskedBounds(object):
    """Object to support reassignment of free parameter bounds.
    """
    def __init__(self, a, m):
        self.a = a
        self.m = m
    
    def __getitem__(self, pos):
        return self.a[self.m[pos]]
    
    def __setitem__(self, pos, value):
        self.a[self.m[pos]] = value
    
    def __len__(self):
        return len(self.m)

class ProductJointPrior(JointPrior):
    """Product of two independent priors.
    
    Parameters
    ----------
    p1, p2: :py:class:`JointPrior` instances
        The two priors to merge.
    """
    def __init__(self, p1, p2):
        if not isinstance(p1, JointPrior) or not isinstance(p2, JointPrior):
            raise TypeError("Both arguments to ProductPrior must be instances "
                            "of type JointPrior!")
        self.p1 = p1
        self.p2 = p2
    
    @property
    def bounds(self):
        return CombinedBounds(self.p1.bounds, self.p2.bounds)
    
    @bounds.setter
    def bounds(self, v):
        num_p1_bounds = len(self.p1.bounds)
        self.p1.bounds = v[:num_p1_bounds]
        self.p2.bounds = v[num_p1_bounds:]

    def __call__(self, theta):
        """Evaluate the prior log-PDF at the given values of the hyperparameters, theta.
        
        The log-PDFs of the two priors are summed.
        
        Parameters
        ----------
        theta : array-like, (`num_params`,)
            The hyperparameters to evaluate the log-PDF at.
        """
        p1_num_params = len(self.p1.bounds)
        return self.p1(theta[:p1_num_params]) + self.p2(theta[p1_num_params:])
    
    def random_draw(self, size=None):
        """Draw random samples of the hyperparameters.

        The outputs of the two priors are stacked vertically.
        
        Parameters
        ----------
        size : None, int or array-like, optional
            The number/shape of samples to draw. If None, only one sample is
            returned. Default is None.
        """
        return scipy.vstack((self.p1.random_draw(size=size), self.p2.random_draw(size=size)))

class UniformJointPrior(JointPrior):
    """Uniform prior over the specified bounds.
    
    Parameters
    ----------
    bounds : list of tuples, (`num_params`,)
        The bounds for each of the random variables.
    """
    def __init__(self, bounds):
        self.bounds = bounds
    
    def __call__(self, theta):
        """Evaluate the prior log-PDF at the given values of the hyperparameters, theta.
        
        Parameters
        ----------
        theta : array-like, (`num_params`,)
            The hyperparameters to evaluate the log-PDF at.
        """
        ll = 0
        for v, b in zip(theta, self.bounds):
            if b[0] <= v and v <= b[1]:
                ll += -scipy.log(b[1] - b[0])
            else:
                ll = -scipy.inf
                break
        return ll
    
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
    
    def __call__(self, theta):
        """Evaluate the prior log-PDF at the given values of the hyperparameters, theta.
        
        Parameters
        ----------
        theta : array-like, (`num_params`,)
            The hyperparameters to evaluate the log-PDF at.
        """
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
    
    def __call__(self, theta):
        """Evaluate the prior log-PDF at the given values of the hyperparameters, theta.
        
        Parameters
        ----------
        theta : array-like, (`num_params`,)
            The hyperparameters to evaluate the log-PDF at.
        """
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
        self.univariate_priors = univariate_priors
    
    def __call__(self, theta):
        """Evaluate the prior log-PDF at the given values of the hyperparameters, theta.
        
        Parameters
        ----------
        theta : array-like, (`num_params`,)
            The hyperparameters to evaluate the log-PDF at.
        """
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
        """
        return [p.interval(1) for p in self.univariate_priors]
    
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
    def __init__(self, mu, sigma):
        sigma = scipy.atleast_1d(scipy.asarray(sigma, dtype=float))
        mu = scipy.atleast_1d(scipy.asarray(mu, dtype=float))
        if sigma.shape != mu.shape:
            raise ValueError("sigma and mu must have the same shape!")
        if sigma.ndim != 1:
            raise ValueError("sigma and mu must both be one dimensional!")
        self.sigma = sigma
        self.mu = mu
    
    def __call__(self, theta):
        """Evaluate the prior log-PDF at the given values of the hyperparameters, theta.
        
        Parameters
        ----------
        theta : array-like, (`num_params`,)
            The hyperparameters to evaluate the log-PDF at.
        """
        ll = 0
        for v, s, m in zip(theta, self.sigma, self.mu):
            ll += scipy.stats.norm.logpdf(v, loc=m, scale=s)
        return ll
    
    @property
    def bounds(self):
        """The bounds of the random variable.
        """
        return [scipy.stats.norm.interval(1, loc=m, scale=s) for s, m in zip(self.sigma, self.mu)]
    
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
    def __init__(self, mu, sigma):
        sigma = scipy.atleast_1d(scipy.asarray(sigma, dtype=float))
        mu = scipy.atleast_1d(scipy.asarray(mu, dtype=float))
        if sigma.shape != mu.shape:
            raise ValueError("sigma and mu must have the same shape!")
        if sigma.ndim != 1:
            raise ValueError("sigma and mu must both be one dimensional!")
        self.sigma = sigma
        self.emu = scipy.exp(mu)
    
    def __call__(self, theta):
        """Evaluate the prior log-PDF at the given values of the hyperparameters, theta.
        
        Parameters
        ----------
        theta : array-like, (`num_params`,)
            The hyperparameters to evaluate the log-PDF at.
        """
        ll = 0
        for v, s, em in zip(theta, self.sigma, self.emu):
            ll += scipy.stats.lognorm.logpdf(v, s, loc=0, scale=em)
        return ll
    
    @property
    def bounds(self):
        """The bounds of the random variable.
        """
        return [scipy.stats.lognorm.interval(1, s, loc=0, scale=em) for s, em in zip(self.sigma, self.emu)]
    
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
    def __init__(self, a, b):
        a = scipy.atleast_1d(scipy.asarray(a, dtype=float))
        b = scipy.atleast_1d(scipy.asarray(b, dtype=float))
        if a.shape != b.shape:
            raise ValueError("sigma and mu must have the same shape!")
        if a.ndim != 1:
            raise ValueError("sigma and mu must both be one dimensional!")
        self.a = a
        self.b = b
    
    def __call__(self, theta):
        """Evaluate the prior log-PDF at the given values of the hyperparameters, theta.
        
        Parameters
        ----------
        theta : array-like, (`num_params`,)
            The hyperparameters to evaluate the log-PDF at.
        """
        ll = 0
        for v, a, b in zip(theta, self.a, self.b):
            ll += scipy.stats.gamma.logpdf(v, a, loc=0, scale=1.0 / b)
        return ll
    
    @property
    def bounds(self):
        """The bounds of the random variable.
        """
        return [scipy.stats.gamma.interval(1, a, loc=0, scale=1.0 / b) for a, b in zip(self.a, self.b)]
    
    def random_draw(self, size=None):
        """Draw random samples of the hyperparameters.
        
        Parameters
        ----------
        size : None, int or array-like, optional
            The number/shape of samples to draw. If None, only one sample is
            returned. Default is None.
        """
        return scipy.asarray([scipy.stats.gamma.rvs(a, loc=0, scale=1.0 / b, size=size) for a, b in zip(self.a, self.b)])

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
    
    From itertools documentation.
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

def univariate_envelope_plot(x, mean, std, ax=None, base_alpha=0.375, envelopes=[1, 3], **kwargs):
    """Make a plot of a mean curve with uncertainty envelopes.
    """
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1)
    elif ax == 'gca':
        ax = plt.gca()
    
    l = ax.plot(x, mean, **kwargs)
    color = plt.getp(l[0], 'color')
    e = []
    for i in envelopes:
        e.append(
            ax.fill_between(
                x,
                mean - i * std,
                mean + i * std,
                facecolor=color,
                alpha=base_alpha / i
            )
        )
    return (l, e)

def summarize_sampler(sampler, burn=0, thin=1, ci=0.95):
    r"""Create summary statistics of the flattened chain of the sampler.
    
    The confidence regions are computed from the quantiles of the data.
    
    Parameters
    ----------
    sampler : :py:class:`emcee.EnsembleSampler` instance
        The sampler to summarize the chains of.
    burn : int, optional
        The number of samples to burn from the beginning of the chain. Default
        is 0 (no burn).
    thin : int, optional
        The step size to thin with. Default is 1 (no thinning).
    ci : float, optional
        A number between 0 and 1 indicating the confidence region to compute.
        Default is 0.95 (return upper and lower bounds of the 95% confidence
        interval).
    
    Returns
    -------
    mean : array, (num_params,)
        Mean values of each of the parameters sampled.
    ci_l : array, (num_params,)
        Lower bounds of the `ci*100%` confidence intervals.
    ci_u : array, (num_params,)
        Upper bounds of the `ci*100%` confidence intervals.
    """
    flat_trace = sampler.chain[:, burn::thin, :]
    flat_trace = flat_trace.reshape((-1, flat_trace.shape[2]))
    
    mean = scipy.mean(flat_trace, axis=0)
    cibdry = 100.0 * (1.0 - ci) / 2.0
    ci_l, ci_u = scipy.percentile(flat_trace, [cibdry, 100.0 - cibdry], axis=0)
    
    return (mean, ci_l, ci_u)

def plot_sampler(sampler, labels=None, burn=0, chain_mask=None):
    """Plot the results of MCMC sampler (posterior and chains).
    
    Loosely based on triangle.py.
    
    Parameters
    ----------
    sampler : :py:class:`emcee.EnsembleSampler` instance
        The sampler to plot the chains/marginals of.
    labels : list of str, optional
        The labels to use for each of the free parameters. Default is to leave
        the axes unlabeled.
    burn : int, optional
        The number of samples to burn before making the marginal histograms.
        Default is zero (use all samples).
    chain_mask : (index) array
        Mask identifying the chains to keep before plotting, in case there are
        bad chains. Default is to use all chains.
    """
    
    # Create axes:
    k = sampler.flatchain.shape[1]
    
    if labels is None:
        labels = [''] * k
    
    f = plt.figure()
    gs1 = mplgs.GridSpec(k, k)
    gs2 = mplgs.GridSpec(1, k)
    gs1.update(bottom=0.275, top=0.98)
    gs2.update(bottom=0.1, top=0.2)
    axes = []
    # j is the row, i is the column.
    for j in xrange(0, k + 1):
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
        axes.append(row)
    axes = scipy.asarray(axes)
    
    # Update axes with the data:
    if chain_mask is None:
        chain_mask = scipy.ones(sampler.chain.shape[0], dtype=bool)
    flat_trace = sampler.chain[chain_mask, burn:, :]
    flat_trace = flat_trace.reshape((-1, k))
    
    # j is the row, i is the column.
    for i in xrange(0, k):
        axes[i, i].clear()
        axes[i, i].hist(flat_trace[:, i], bins=50, color='black')
        if i == k - 1:
            axes[i, i].set_xlabel(labels[i])
        if i < k - 1:
            plt.setp(axes[i, i].get_xticklabels(), visible=False)
        plt.setp(axes[i, i].get_yticklabels(), visible=False)
        # for j in xrange(0, i):
        #     axes[j, i].set_visible(False)
        #     axes[j, i].set_frame_on(False)
        for j in xrange(i + 1, k):
            axes[j, i].clear()
            ct, x, y, im = axes[j, i].hist2d(flat_trace[:, i], flat_trace[:, j], bins=50, cmap='gray_r')
            # xmid = 0.5 * (x[1:] + x[:-1])
            # ymid = 0.5 * (y[1:] + y[:-1])
            # axes[j, i].contour(xmid, ymid, ct.T, colors='k')
            if j < k - 1:
                plt.setp(axes[j, i].get_xticklabels(), visible=False)
            if i != 0:
                plt.setp(axes[j, i].get_yticklabels(), visible=False)
            if i == 0:
                axes[j, i].set_ylabel(labels[j])
            if j == k - 1:
                axes[j, i].set_xlabel(labels[i])
        axes[-1, i].clear()
        axes[-1, i].plot(sampler.chain[:, :, i].T)
        axes[-1, i].axvline(burn, color='r', linewidth=3)
        axes[-1, i].set_ylabel(labels[i])
        axes[-1, i].set_xlabel('step')
    
    f.canvas.draw()
