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

"""Provides the base :py:class:`GaussianProcess` class.
"""

from __future__ import division

from .error_handling import GPArgumentError
from .kernel import Kernel, ZeroKernel
from .utils import wrap_fmin_slsqp

import scipy
import scipy.linalg
import scipy.optimize
import scipy.stats
import numpy.random
import sys
import warnings

class GaussianProcess(object):
    r"""Gaussian process.
    
    If called with one argument, an untrained Gaussian process is
    constructed and training data must be added with the :py:meth:`add_data` method.
    If called with the optional keywords, the values given are used as the
    training data. It is always possible to add additional training data
    with :py:meth:`add_data`.
    
    Note that the attributes have no write protection, but you should always
    add data with :py:meth:`add_data` to ensure internal consistency.
    
    Parameters
    ----------
    k : :py:class:`~gptools.kernel.core.Kernel` instance
        Kernel instance corresponding to the desired noise-free
        covariance kernel of the Gaussian process. The noise is handled
        separately either through specification of `err_y`, or in a
        separate kernel. This allows noise-free predictions when needed.
    
    noise_k : :py:class:`~gptools.kernel.core.Kernel` instance
        Kernel instance corresponding to the noise portion of the
        desired covariance kernel of the Gaussian process. Note that you
        DO NOT need to specify this if the extent of the noise you want
        to represent is contained in `err_y` (or if your data are
        noiseless). Default value is None, which results in the
        :py:class:`~gptools.kernel.noise.ZeroKernel` (noise specified elsewhere
        or not present).
    
    standardize : bool
        Flag for whether or not all internal calculations should be done with
        standardized variables (right now only y is standarized):

        .. math::

            Z = \frac{y - \mu}{\sigma}

        Notice that this will change the interpretation of scale parameters
        :math:`\sigma` to be normalized scales :math:`\sigma/\sigma_y`.
        Regardless of the state of this flag, :py:meth:`predict` and
        :py:meth:`draw_sample` will always return in real units. Default value
        is False (do internal calculations in real units).
    
    NOTE
        The following are all passed to :py:meth:`add_data`, refer to its docstring.
    
    X : :py:class:`Matrix` or other Array-like, (`M`, `N`), optional
        `M` training input values of dimension `N`. Default value is None (no
        training data).
        
    y : :py:class:`Array` or other Array-like, (`M`,), optional
        `M` training target values. Default value is None (no training data).
        
    err_y : :py:class:`Array` or other Array-like, (`M`,), optional
        Error (given as standard deviation) in the `M` training target values.
        Default value is 0 (noiseless observations).
    
    Attributes
    ----------
    k : :py:class:`~gptools.kernel.core.Kernel` instance
        The non-noise portion of the covariance kernel.
    noise_k : :py:class:`~gptools.kernel.core.Kernel` instance
        The noise portion of the covariance kernel.
    standardize : bool
        True if internal calculations are done with standardized variables, False
        if real units are used.
    X : :py:class:`Matrix`, (`M`, `N`)
        The `M` training input values, each of which is of dimension `N`.
    y : :py:class:`Array`, (`M`,)
        The `M` training target values.
    err_y : :py:class:`Array`, (`M`,)
        The error in the `M` training input values.
    n : :py:class:`Matrix`, (`M`, `N`)
        The orders of derivatives that each of the M training points represent, indicating the order of derivative with respect to each of the `N` dimensions.
    K_up_to_date : bool
        True if no data have been added since the last time the internal state was updated with a call to :py:meth:`compute_K_L_alpha_ll`.
    K : :py:class:`Matrix`, (`M`, `M`)
        Covariance matrix between all of the training inputs.
    noise_K : :py:class:`Matrix`, (`M`, `M`)
        Noise portion of the covariance matrix between all of the training inputs. Only includes the noise from :py:attr:`noise_k`, not from :py:attr:`err_y`.
    L : :py:class:`Matrix`, (`M`, `M`)
        Cholesky decomposition of the combined covariance matrix between all of the training inputs.
    alpha : :py:class:`Matrix`, (`M`, 1)
        Solution to :math:`K\alpha=y`.
    ll : float
        Log-likelihood of the data given the model.
    
    Raises
    ------
    GPArgumentError
        Gave `X` but not `y` (or vice versa).
    ValueError
        Training data rejected by :py:meth:`add_data`.
    
    See Also
    --------
    add_data : Used to process `X`, `y`, `err_y` and to add data to the process.
    """
    def __init__(self, k, noise_k=None, standardize=False, X=None, y=None, err_y=0):
        if standardize:
            raise ValueError("Keyword standardize is not supported at this point!")
        if not isinstance(k, Kernel):
            raise TypeError("Argument k must be an instance of Kernel when "
                            "constructing GaussianProcess!")
        if noise_k is None:
            noise_k = ZeroKernel(k.num_dim)
        else:
            if not isinstance(noise_k, Kernel):
                raise TypeError("Keyword noise_k must be an instance of Kernel "
                                "when constructing GaussianProcess!")

        self.standardize = standardize
        self.k = k
        self.noise_k = noise_k
        self.y = scipy.array([], dtype=float)
        self.X = None
        self.err_y = scipy.array([], dtype=float)
        self.n = None
        if X is not None:
            if y is None:
                raise GPArgumentError("Must pass both X and y when "
                                      "constructing GaussianProcess!")
            else:
                self.add_data(X, y, err_y=err_y, n=0)
        elif X is None and y is not None:
            raise GPArgumentError("Must pass both X and y when constructing "
                                  "GaussianProcess!")
        else:
            self.K_up_to_date = False
    
    @property
    def num_dim(self):
        """The number of dimensions of the input data.
        
        Returns
        -------
        num_dim: int
            The number of dimensions of the input data as defined in the kernel.
        """
        return self.k.num_dim
    
    def add_data(self, X, y, err_y=0, n=0):   
        """Add data to the training data set of the GaussianProcess instance.
        
        Parameters
        ----------
        X : :py:class:`Matrix` or other Array-like, (`M`, `N`)
            `M` training input values of dimension `N`.
        y : :py:class:`Array` or other Array-like, (`M`,)
            `M` training target values.
        err_y : :py:class:`Array` or other Array-like (`M`,) or scalar float, optional
            Non-negative values only. Error given as standard deviation) in the
            `M` training target values. If `err_y` is a scalar, the data set is
            taken to be homoscedastic (constant error). Otherwise, the length
            of `err_y` must equal the length of `y`. Default value is 0
            (noiseless observations).
        n : :py:class:`Matrix` or other Array-like (`M`, `N`) or scalar float, optional
            Non-negative integer values only. Degree of derivative for each
            training target. If `n` is a scalar it is taken to be the value for
            all points in `y`. Otherwise, the length of n must equal the length
            of `y`. Default value is 0 (observation of target value). If
            non-integer values are passed, they will be silently rounded.
        
        Raises
        ------
        ValueError
            Bad shapes for any of the inputs, negative values for `err_y` or `n`.
        """
        # Verify y has only one non-trivial dimension:
        try:
            iter(y)
        except TypeError:
            y = scipy.array([y], dtype=float)
        else:
            y = scipy.asarray(y, dtype=float)
            if len(y.shape) != 1:
                raise ValueError("Training targets y must have only one "
                                 "dimension with length greater than one! Shape "
                                 "of y given is %s" % (y.shape,))
        
        # Handle scalar error or verify shape of array error matches shape of y:
        try:
            iter(err_y)
        except TypeError:
            err_y = err_y * scipy.ones_like(y, dtype=float)
        else:
            err_y = scipy.asarray(err_y, dtype=float)
            if err_y.shape != y.shape:
                raise ValueError("When using array-like err_y, shape must match "
                                 "shape of y! Shape of err_y given is %s, shape "
                                 "of y given is %s." % (err_y.shape, y.shape))
        if (err_y < 0).any():
            raise ValueError("All elements of err_y must be non-negative!")
        
        # Handle scalar derivative orders or verify shape of array derivative
        # orders matches shape of y:
        try:
            iter(n)
        except TypeError:
            n = n * scipy.asmatrix(scipy.ones((len(y), self.num_dim), dtype=int))
        else:
            n = scipy.asmatrix(n, dtype=int)
            # Correct single-dimension inputs:
            if self.num_dim == 1 and n.shape[1] != 1:
                n = n.T
            if n.shape != (len(y), self.num_dim):
                raise ValueError("When using array-like n, shape must be "
                                 "(len(y), k.num_dim)! Shape of n given is %s, "
                                 "shape of y given is %s and num_dim=%d." % (n.shape, y.shape, self.num_dim))
        if (n < 0).any():
            raise ValueError("All elements of n must be non-negative integers!")
        
        # Handle scalar training input or convert array input into matrix.
        X = scipy.asmatrix(X, dtype=float)
        # Correct single-dimension inputs:
        if self.num_dim == 1 and X.shape[0] == 1:
            X = X.T
        if X.shape != (len(y), self.num_dim):
            raise ValueError("Shape of training inputs must be (len(y), k.num_dim)! "
                             "X given has shape %s, shape of "
                             "y is %s and num_dim=%d." % (X.shape, y.shape, self.num_dim))
        
        if self.X is None:
            self.X = X
        else:
            self.X = scipy.vstack((self.X, X))
        self.y = scipy.append(self.y, y)
        self.err_y = scipy.append(self.err_y, err_y)
        if self.n is None:
            self.n = n
        else:
            self.n = scipy.vstack((self.n, n))
        self.K_up_to_date = False
    
    def compute_Kij(self, Xi, Xj, ni, nj, noise=False, hyper_deriv=None):
        r"""Compute covariance matrix between datasets `Xi` and `Xj`.
        
        Specify the orders of derivatives at each location with the `ni`, `nj`
        arrays. The `include_noise` flag is passed to the covariance kernel to
        indicate whether noise is to be included (i.e., for evaluation of
        :math:`K+\sigma I` versus :math:`K_*`).
        
        If `Xj` is None, the symmetric matrix :math:`K(X, X)` is formed.
        
        Note that type and dimension checking is NOT performed, as it is assumed
        the data are from inside the instance and have hence been sanitized by
        :py:meth:`add_data`.
        
        Parameters
        ----------
        Xi : :py:class:`Matrix`, (`M`, `N`)
            `M` input values of dimension `N`.
        Xj : :py:class:`Matrix`, (`P`, `N`)
            `P` input values of dimension `N`.
        ni : :py:class:`Array`, (`M`,), non-negative integers
            `M` derivative orders with respect to the `Xi` coordinates.
        nj : :py:class:`Array`, (`P`,), non-negative integers
            `P` derivative orders with respect to the `Xj` coordinates.
        noise : bool, optional
            If True, uses the noise kernel, otherwise uses the regular kernel.
            Default is False (use regular kernel).
        hyper_deriv : None or non-negative int
            Index of the hyperparameter to compute the first derivative with
            respect to. If None, no derivatives are taken. Default is None (no
            hyperparameter derivatives).
                
        Returns
        -------
        Kij : :py:class:`Matrix`, (`M`, `P`)
            Covariance matrix between `Xi` and `Xj`.
        """
        if not noise:
            k = self.k
        else:
            k = self.noise_k
        
        if Xj is None:
            symmetric = True
            Xj = Xi
            nj = ni
        else:
            symmetric = False
        
        # This technically doesn't take advantage of the symmetric case. Might
        # be worth trying to do that at some point, but this is vastly superior
        # to the double for loop implementation for which using symmetry is easy.
        Xi_tile = scipy.repeat(Xi, Xj.shape[0], axis=0)
        ni_tile = scipy.repeat(ni, Xj.shape[0], axis=0)
        Xj_tile = scipy.tile(Xj, (Xi.shape[0], 1))
        nj_tile = scipy.tile(nj, (Xi.shape[0], 1))
        Kij = k(Xi_tile, Xj_tile, ni_tile, nj_tile, hyper_deriv=hyper_deriv, symmetric=symmetric)
        Kij = scipy.asmatrix(scipy.reshape(Kij, (Xi.shape[0], -1)))
        
        return Kij
    
    def compute_K_L_alpha_ll(self, diag_factor=1e2):
        r"""Compute `K`, `L`, `alpha` and log-likelihood according to the first part of Algorithm 2.1 in R&W.
        
        Computes `K` and the noise portion of `K` using :py:meth:`compute_Kij`,
        computes `L` using :py:func:`scipy.linalg.cholesky`, then computes
        `alpha` as `L.T\\(L\\y)`.
        
        Only does the computation if :py:attr:`K_up_to_date` is False --
        otherwise leaves the existing values.
        
        Parameters
        ----------
        diag_factor : float, optional
            Factor of :py:attr:`sys.float_info.epsilon` which is added to
            the diagonal of the total `K` matrix to improve the stability of
            the Cholesky decomposition. If you are having issues, try increasing
            this by a factor of 10 at a time. Default is 1e2.
        """
        if not self.K_up_to_date:
            if self.standardize:
                # TODO: Implement standardization on X!
                # TODO: This does not handle the derivatives properly!
                self.mu_y = scipy.mean(self.y)
                self.std_y = scipy.std(self.y, ddof=1)
                self.y_s = (self.y - self.mu_y) / self.std_y
                y = self.y_s
                self.err_y_s = self.err_y / self.std_y
                err_y = self.err_y_s
            else:
                y = self.y
                err_y = self.err_y
            self.K = self.compute_Kij(self.X, None, self.n, None, noise=False)
            self.noise_K = self.compute_Kij(self.X, None, self.n, None, noise=True)
            K_tot = (self.K +
                     scipy.diag(err_y**2.0) +
                     self.noise_K +
                     diag_factor * sys.float_info.epsilon * scipy.eye(len(y)))
            try:
                self.L = scipy.matrix(
                    scipy.linalg.cholesky(
                        K_tot,
                        lower=True,
                        check_finite=False
                    )
                )
            except TypeError:
                # Catch lack of check_finite in older scipy:
                self.L = scipy.matrix(
                    scipy.linalg.cholesky(
                        K_tot,
                        lower=True
                    )
                )
            # Convert the array output to a matrix since scipy treats arrays
            # as row vectors:
            try:
                self.alpha = scipy.linalg.solve_triangular(
                    self.L.T,
                    scipy.linalg.solve_triangular(
                        self.L,
                        scipy.asmatrix(y).T,
                        lower=True,
                        check_finite=False
                    ),
                    lower=False,
                    check_finite=False
                )
            except TypeError:
                self.alpha = scipy.linalg.solve_triangular(
                    self.L.T,
                    scipy.linalg.solve_triangular(
                        self.L,
                        scipy.asmatrix(y).T,
                        lower=True
                    ),
                    lower=False
                )
            self.ll = (-0.5 * scipy.asmatrix(y) * self.alpha -
                       scipy.log(scipy.diag(self.L)).sum() - 
                       0.5 * len(y) * scipy.log(2.0 * scipy.pi))[0, 0]
            # Apply hyperpriors:
            # TODO: Is there a more pythonic way of doing this?
            for p, is_log, theta in zip(list(self.k.hyperpriors) + list(self.noise_k.hyperpriors),
                                        list(self.k.is_log) + list(self.noise_k.is_log),
                                        list(self.k.params) + list(self.noise_k.params)):
                if is_log:
                    self.ll += p(theta)
                else:
                    self.ll += scipy.log(p(theta))
            self.K_up_to_date = True
    
    def update_hyperparameters(self, new_params, return_jacobian=False):
        """Update the kernel's hyperparameters to the new parameters.
        
        This will call :py:meth:`compute_K_L_alpha_ll` to update the state
        accordingly.
        
        Parameters
        ----------
        new_params : :py:class:`Array` or other Array-like, length dictated by kernel
            New parameters to use.
        return_jacobian : bool, optional
            If True, the return is (`ll`, `jac`). Otherwise, return is `ll`
            only and the execution is faster. Default is False (do not
            compute Jacobian).
        
        Returns
        -------
        -1*ll : float
            The updated log likelihood.
        -1*jac : :py:class:`Array`, length equal to the number of parameters
            The derivative of `ll` with respect to each of the parameters, in
            order. Only computed and returned if `return_jacobian` is True.
        """
        self.k.set_hyperparams(new_params[:len(self.k.free_params)])
        self.noise_k.set_hyperparams(new_params[len(self.k.free_params):])
        self.K_up_to_date = False
        self.compute_K_L_alpha_ll()
        if not return_jacobian:
            return -1 * self.ll
        else:
            # Doesn't handle noise!
            aaKI = self.alpha * self.alpha.T - self.K.I
            jac = scipy.zeros_like(self.k.free_params, dtype=float)
            for i in xrange(0, len(jac)):
                # TODO: Put in noise
                dKijdHP = self.compute_Kij(self.X, None, self.n, None, hyper_deriv=i)
                # TODO: Compare timing between doing the full product and
                # extracting only the trace.
                jac[i] = 0.5 * scipy.trace(aaKI * dKijdHP)
            return (-1 * self.ll, -1 * jac)

    def optimize_hyperparameters(self, method='SLSQP', opt_kwargs={}, verbose=False):
        r"""Optimize the hyperparameters by maximizing the log likelihood.
        
        Leaves the :py:class:`GaussianProcess` instance in the optimized state.
        
        If :py:func:`scipy.optimize.minimize` is not available (i.e., if your
        :py:mod:`scipy` version is older than 0.11.0) then :py:func:`fmin_slsqp`
        is used independent of what you set for the `method` keyword.
        
        Parameters
        ----------
        method : str, optional
            The method to pass to :py:func:`scipy.optimize.minimize`.
            Refer to that function's docstring for valid options. Default
            is 'SLSQP'. See note above about behavior with older versions of
            :py:mod:`scipy`.
        opt_kwargs : dict, optional
            Dictionary of extra keywords to pass to
            :py:func:`scipy.optimize.minimize`. Refer to that function's docstring for
            valid options. Note that if you use `jac` = True (i.e., optimization
            function returns Jacobian) you should also set `args` = (True,) to
            tell :py:meth:`update_hyperparameters` to compute and return the
            Jacobian. Default is: {}.
        verbose : bool, optional
            Whether or not the output should be verbose. If
            True, the entire :py:class:`Result` object from
            :py:func:`scipy.optimize.minimize` is printed. If False, status
            information is only printed if the `success` flag from
            :py:func:`minimize` is False. Default is False.
        """
        if opt_kwargs is None:
            #opt_kwargs = {'args': (False,),
            #              'bounds': scipy.concatenate((self.k.free_param_bounds, self.noise_k.free_param_bounds)),
            #              'jac': None}
            opt_kwargs = {}
        else:
            opt_kwargs = dict(opt_kwargs)
        # TODO: Add ability to do random starts to avoid local minima.
        if 'method' in opt_kwargs:
            warnings.warn("Use of keyword 'method' in opt_kwargs is not allowed, "
                          "and is being ignored. Use the 'method' keyword for "
                          "optimize_hyperparameters instead.",
                          RuntimeWarning)
            opt_kwargs.pop('method')
        try:
            res = scipy.optimize.minimize(self.update_hyperparameters,
                                          scipy.concatenate((self.k.free_params, self.noise_k.free_params)),
                                          method=method,
                                          **opt_kwargs)
        except AttributeError:
            warnings.warn("scipy.optimize.minimize not available, defaulting to fmin_slsqp.",
                          RuntimeWarning)
            res = wrap_fmin_slsqp(self.update_hyperparameters,
                                   scipy.concatenate((self.k.free_params, self.noise_k.free_params)),
                                   opt_kwargs=opt_kwargs)
            
        self.update_hyperparameters(res.x, return_jacobian=False)
        if verbose:
            print(res)
        if not res.success:
            warnings.warn("Solver %s reports failure, selected hyperparameters "
                          "are likely NOT optimal. Status: %d, Message: '%s'"
                          % (method, res.status, res.message),
                          RuntimeWarning)
    
    def predict(self, Xstar, n=0, noise=False, return_cov=True):
        """Predict the mean and covariance at the inputs `Xstar`.
        
        The order of the derivative is given by `n`. The keyword `noise` sets
        whether or not noise is included in the prediction.
        
        Parameters
        ----------
        Xstar : :py:class:`Array` or other Array-like, (`M`, `N`)
            `M` test input values of dimension `N`.
        n : :py:class:`Matrix` or other Array-like, (`M`, `N`) or scalar, non-negative int, optional
            Order of derivative to predict (0 is the base quantity). If `n` is
            scalar, the value is used for all points in `Xstar`. If non-integer
            values are passed, they will be silently rounded. Default is 0
            (return base quantity).
        noise : bool, optional
            Whether or not noise should be included in the covariance. Default
            is False (no noise in covariance).
        return_cov : bool, optional
            Set to True to compute and return the covariance matrix for the
            predictions, False to skip this step. Default is True (return tuple
            of (`mean`, `cov`)).
        
        Returns
        -------
        mean : :py:class:`Array`, (`M`,)
            Predicted GP mean.
        covariance : :py:class:`Matrix`, (`M`, `M`)
            Predicted covariance matrix, only returned if `return_cov` is True.
        
        Raises
        ------
        ValueError
            If `n` is not consistent with the shape of `Xstar` or is not entirely
            composed of non-negative integers.
        """
        # Process Xstar:
        Xstar = scipy.asmatrix(Xstar, dtype=float)
        # Handle 1d x case where array is passed in:
        if self.num_dim == 1 and Xstar.shape[0] == 1:
            Xstar = Xstar.T
        if Xstar.shape[1] != self.num_dim:
            raise ValueError("Second dimension of Xstar must be equal to "
                             "self.num_dim! Shape of Xstar given is %s, "
                             "num_dim is %d." % (Xstar.shape, self.num_dim))
        # Process n:
        try:
            iter(n)
        except TypeError:
            n = n * scipy.asmatrix(scipy.ones(Xstar.shape, dtype=int))
        else:
            n = scipy.asmatrix(n, dtype=int)
            if self.num_dim == 1 and n.shape[0] == 1:
                n = n.T
            if n.shape != Xstar.shape:
                raise ValueError("When using array-like n, shape must match "
                                 "shape of Xstar! Shape of n given is %s, "
                                 "shape of Xstar given is %s." % (n.shape, Xstar.shape))
        if (n < 0).any():
            raise ValueError("All elements of n must be non-negative integers!")
        
        self.compute_K_L_alpha_ll()
        Kstar = self.compute_Kij(self.X, Xstar, self.n, n)
        if noise:
            Kstar = Kstar + self.compute_Kij(self.X, Xstar, self.n, n, noise=True)
        mean = Kstar.T * self.alpha
        if self.standardize:
            mean = mean * self.std_y + self.mu_y
        if return_cov:
            try:
                v = scipy.asmatrix(
                    scipy.linalg.solve_triangular(self.L, Kstar, lower=True, check_finite=False)
                )
            except TypeError:
                # Handle older versions of scipy:
                v = scipy.asmatrix(
                    scipy.linalg.solve_triangular(self.L, Kstar, lower=True)
                )
            Kstarstar = self.compute_Kij(Xstar, None, n, None)
            if noise:
                Kstarstar = Kstarstar + self.compute_Kij(Xstar, None, n, None, noise=True)
            covariance = Kstarstar - v.T * v
            if self.standardize:
                covariance = covariance * self.std_y**2.0

            return (mean, covariance)
        else:
            return mean
    
    def compute_ll_matrix(self, bounds, num_pts):
        """Compute the log likelihood over the (free) parameter space.
        
        Parameters
        ----------
        bounds : 2-tuple or list of 2-tuples with length equal to the number of free parameters
            Bounds on the range to use for each of the parameters. If a single
            2-tuple is given, it will be used for each of the parameters.
        num_pts : int or list of ints with length equal to the number of free parameters
            If a single int is given, it will be used for each of the parameters.
        
        Returns
        -------
            ll_vals : :py:class:`Array`
                The log likelihood for each of the parameter possibilities.
            param_vals : List of :py:class:`Array`
                The parameter values used.
        """
        present_free_params = scipy.concatenate((self.k.free_params, self.noise_k.free_params))
        bounds = scipy.atleast_2d(scipy.asarray(bounds, dtype=float))
        if bounds.shape[1] != 2:
            raise ValueError("Argument bounds must have shape (n, 2)!")
        # If bounds is a single tuple, repeat it for each free parameter:
        if bounds.shape[0] == 1:
            bounds = scipy.tile(bounds, (len(present_free_params), 1))
        # If num_pts is a single value, use it for all of the parameters:
        try:
            iter(num_pts)
        except TypeError:
            num_pts = num_pts * scipy.ones(bounds.shape[0], dtype=int)
        else:
            num_pts = scipy.asarray(num_pts, dtype=int)
            if len(num_pts) != len(present_free_params):
                raise ValueError("Length of num_pts must match the number of free parameters of kernel!")
        
        # Form arrays to evaluate parameters over:
        param_vals = []
        for k in xrange(0, len(present_free_params)):
            param_vals.append(scipy.linspace(bounds[k, 0], bounds[k, 1], num_pts[k]))
        ll_vals = self._compute_ll_matrix(0, param_vals, num_pts)
        
        # Reset the parameters to what they were before:
        self.update_hyperparameters(scipy.asarray(present_free_params, dtype=float))
        
        return (ll_vals, param_vals)
    
    def _compute_ll_matrix(self, idx, param_vals, num_pts):
        """Recursive helper function for compute_ll_matrix.
        
        Parameters
        ----------
        idx : int
            The index of the parameter for this layer of the recursion to
            work on. `idx` == len(`num_pts`) is the base case that terminates
            the recursion.
        param_vals : List of :py:class:`Array`
            List of arrays of parameter values. Entries in the slots 0:`idx` are
            set to scalars by the previous levels of recursion.
        num_pts : :py:class:`Array`
            The numbers of points for each parameter.
        
        Returns
        -------
        vals : :py:class:`Array`
            The log likelihood for each of the parameter possibilities at lower
            levels.
        """
        if idx >= len(num_pts):
            # Base case: All entries in param_vals should be scalars:
            return -1.0 * self.update_hyperparameters(scipy.asarray(param_vals, dtype=float))
        else:
            # Recursive case: call _compute_ll_matrix for each entry in param_vals[idx]:
            vals = scipy.zeros(num_pts[idx:], dtype=float)
            for k in xrange(0, len(param_vals[idx])):
                specific_param_vals = list(param_vals)
                specific_param_vals[idx] = param_vals[idx][k]
                vals[k] = self._compute_ll_matrix(idx + 1, specific_param_vals, num_pts)
            return vals
    
    def draw_sample(self, Xstar, n=0, noise=False, num_samp=1, rand_vars=None,
                    rand_type='standard normal', diag_factor=1e3, method='cholesky',
                    num_eig=None):
        """Draw a sample evaluated at the given points `Xstar`.
        
        Parameters
        ----------
        Xstar : :py:class:`Matrix` or other Array-like, (`M`, `N`)
            `M` test input values of dimension `N`.
        n : :py:class:`Matrix` or other Array-like, (`M`, `N`) or scalar, non-negative int, optional
            Derivative order to evaluate at. Default is 0 (evaluate value).
        noise : bool, optional
            Whether or not to include the noise components of the kernel in the
            sample. Default is False (no noise in samples).
        num_samp : Positive int, optional
            Number of samples to draw. Default is 1. Cannot be used in
            conjunction with `rand_vars`: If you pass both `num_samp` and
            `rand_vars`, `num_samp` will be silently ignored.
        rand_vars : :py:class:`Matrix` or other Array-like (`M`, `P`), optional
            Vector of random variables :math:`u` to use in constructing the
            sample :math:`y_* = f_* + Lu`, where :math:`K=LL^T`. If None,
            values will be produced using :py:func:`numpy.random.multivariate_normal`.
            This allows you to use pseudo/quasi random numbers generated by
            an external routine. Default is None (use :py:func:`multivariate_normal`
            directly).
        rand_type : {'standard normal', 'uniform'}, optional
            Type of distribution the inputs are given with.
            
                * 'standard normal': Standard (`mu` = 0, `sigma` = 1) normal
                  distribution (this is the default)
                * 'uniform': Uniform distribution on [0, 1). In this case
                  the required Gaussian variables are produced with inversion.
                  
        diag_factor : float, optional
            Number (times machine epsilon) added to the diagonal of the
            covariance matrix prior to computing its Cholesky decomposition.
            This is necessary as sometimes the decomposition will fail because,
            to machine precision, the matrix appears to not be positive definite.
            If you are getting errors from :py:func:`scipy.linalg.cholesky`, try increasing
            this an order of magnitude at a time. This parameter only has an
            effect when using rand_vars. Default value is 1e3. 
        method : {'cholesky', 'eig'}, optional
            Method to use for constructing the matrix square root. Default is
            'cholesky' (use lower-triangular Cholesky decomposition).
            
                * 'cholesky': Perform Cholesky decomposition on the covariance
                  matrix: :math:`K=LL^T`, use :math:`L` as the matrix square
                  root.
                * 'eig': Perform an eigenvalue decomposition on the covariance
                  matrix: :math:`K=Q \\Lambda Q^{-1}`, use :math:`Q\\Lambda^{1/2}`
                  as the matrix square root.
        num_eig : int or None, optional
            Number of eigenvalues to compute. Can range from 1 to `M` (the
            number of test points). If it is None, then all eigenvalues are
            computed. Default is None (compute all eigenvalues). This keyword
            only has an effect if `method` is 'eig'.
        
        Returns
        -------
            samples : :py:class:`Array` (`M`, `P`) or (`M`, `num_samp`)
                Samples evaluated at the `M` points.
        
        Raises
        ------
        ValueError
            If rand_type or method is invalid.
        """
        # All of the input processing for Xstar and n will be done in here:
        mean, cov = self.predict(Xstar, n=n, noise=noise)
        if rand_vars is None and method != 'eig':
            return scipy.asarray(numpy.random.multivariate_normal(scipy.asarray(mean).flatten(), cov, num_samp)).T
        else:
            if num_eig is None or num_eig > len(mean):
                num_eig = len(mean)
            elif num_eig < 1:
                num_eig = 1
            if rand_vars is None:
                rand_vars = numpy.random.standard_normal((num_eig, num_samp))
            valid_types = ('standard normal', 'uniform')
            if rand_type not in valid_types:
                raise ValueError("rand_type %s not recognized! Valid options are: %s." % (rand_type, valid_types,))
            if rand_type == 'uniform':
                rand_vars = scipy.stats.norm.ppf(rand_vars)
            
            # TODO: Should probably do some shape-checking first...
            
            if method == 'cholesky':
                L = scipy.asmatrix(scipy.linalg.cholesky(cov + diag_factor * sys.float_info.epsilon * scipy.eye(cov.shape[0]),
                                                         lower=True,
                                                         check_finite=False),
                                   dtype=float)
            elif method == 'eig':
                # TODO: Add support for specifying cutoff eigenvalue!
                # Not technically lower triangular, but we'll keep the name L:
                eig, Q = scipy.linalg.eigh(cov + diag_factor * sys.float_info.epsilon * scipy.eye(cov.shape[0]),
                                           eigvals=(len(mean) - 1 - (num_eig - 1), len(mean) - 1))
                Lam_1_2 = scipy.asmatrix(scipy.diag(scipy.sqrt(eig)))
                L = scipy.asmatrix(Q) * Lam_1_2
            else:
                raise ValueError("method %s not recognized!" % (method,))
            return mean + L * scipy.asmatrix(rand_vars[:num_eig, :], dtype=float)

class Constraint(object):
    """Implements an inequality constraint on the value of the mean or its derivatives.
    
    Provides a callable such as can be passed to SLSQP or COBYLA to implement
    the constraint when using :py:func:`scipy.optimize.minimize`.
    
    The function defaults implement a constraint that forces the mean value to
    be positive everywhere.
    
    Parameters
    ----------
    gp : :py:class:`GaussianProcess`
        The :py:class:`GaussianProcess` instance to create the constraint on.
    boundary_val : float, optional
        Boundary value for the constraint. For `type_` = 'gt', this is the lower
        bound, for `type_` = 'lt', this is the upper bound. Default is 0.0.
    n : non-negative int, optional
        Derivative order to evaluate. Default is 0 (value of the mean). Note
        that non-int values are silently cast to int.
    loc : {'min', 'max'}, float or Array-like of float (`num_dim`,), optional
        Which extreme of the mean to use, or location to evaluate at.
        
        * If 'min', the minimum of the mean (optionally over `bounds`) is used.
        * If 'max', the maximum of the mean (optionally over `bounds`) is used.
        * If a float (valid for `num_dim` = 1 only) or Array of float, the mean
          is evaluated at the given X value.
        
        Default is 'min' (use function minimum).
    type_ : {'gt', 'lt'}, optional
        What type of inequality constraint to implement.
        
        * If 'gt', a greater-than-or-equals constraint is used.
        * If 'lt', a less-than-or-equals constraint is used.
        
        Default is 'gt' (greater-than-or-equals).
    bounds : 2-tuple of float or 2-tuple Array-like of float (`num_dim`,) or None, optional
        Bounds to use when `loc` is 'min' or 'max'.
        
        * If None, the bounds are taken to be the extremes of the training data.
          For multivariate data, "extremes" essentially means the smallest
          hypercube oriented parallel to the axes that encapsulates all of the
          training inputs. (I.e., ``(gp.X.min(axis=0), gp.X.max(axis=0))``)
        * If `bounds` is a 2-tuple, then this is used as (`lower`, `upper`)
          where lower` and `upper` are Array-like with dimensions (`num_dim`,).
        * If `num_dim` is 1 then `lower` and `upper` can be scalar floats.
        
        Default is None (use extreme values of training data).
    
    Raises
    ------
    TypeError
        If `gp` is not an instance of :py:class:`GaussianProcess`.
    ValueError
        If `n` is negative.
    ValueError
        If `loc` is not 'min', 'max' or an Array-like of the correct dimensions.
    ValueError
        If `type_` is not 'gt' or 'lt'.
    ValueError
        If `bounds` is not None or length 2 or if the elements of bounds don't
        have the right dimensions.
    """
    def __init__(self, gp, boundary_val=0.0, n=0, loc='min', type_='gt', bounds=None):
        if not isinstance(gp, GaussianProcess):
            raise TypeError("Argument gp must be an instance of GaussianProcess.")
        self.gp = gp
        self.boundary_val = boundary_val
        self.n = int(n)
        if self.n < 0:
            raise ValueError("n must be a non-negative int!")
        
        if loc in ('min', 'max'):
            self.loc = loc
        else:
            try:
                iter(loc)
            except TypeError:
                if self.gp.num_dim == 1:
                    self.loc = scipy.asarray([loc], dtype=float)
                else:
                    raise ValueError("Argument loc must be 'min', 'max' or an array of length %d" % self.gp.num_dim)
            else:
                loc = scipy.asarray(loc, dtype=float)
                if loc.shape == (self.gp.num_dim,):
                    self.loc = loc
                else:
                    raise ValueError("Argument loc must be 'min', 'max' or have length %d" % self.gp.num_dim)
        
        if type_ in ('gt', 'lt'):
            self.type_ = type_
        else:
            raise ValueError("Argument type_ must be 'gt' or 'lt'.")
        
        if bounds is None:
            bounds = (scipy.asarray(self.gp.X.min(axis=0), dtype=float).flatten(),
                      scipy.asarray(self.gp.X.max(axis=0), dtype=float).flatten())
        else:
            bounds = list(bounds)
            if len(bounds) != 2:
                raise ValueError("Argument bounds must have length 2!")
            for k in xrange(0, len(bounds)):
                try:
                    iter(bounds[k])
                except TypeError:
                    if self.gp.num_dim == 1:
                        bounds[k] = scipy.asarray([bounds[k]], dtype=float)
                    else:
                        raise ValueError("Each element in argument bounds must have length %d" % self.gp.num_dim)
                else:
                    bounds[k] = scipy.asarray(bounds[k], dtype=float)
                    if bounds[k].shape != (self.gp.num_dim,):
                        raise ValueError("Each element in argument bounds must have length %d" % self.gp.num_dim)
        # Unfold bounds into the shape needed by minimize:
        self.bounds = zip(bounds[0], bounds[1])
    
    def __call__(self, params):
        """Returns a non-negative number if the constraint is satisfied.
        
        Parameters
        ----------
        params : Array-like, length dictated by kernel
            New parameters to use.
        
        Returns
        -------
        val : float
            Value of the constraint. :py:class:`minimize` will attempt to keep
            this non-negative.
        """
        self.gp.update_hyperparameters(params)
        if self.loc not in ('min', 'max'):
            val = self.gp.predict(self.loc, n=self.n, return_cov=False)
        else:
            if self.loc == 'max':
                factor = -1.0
            else:
                factor = 1.0
            
            try:
                res = scipy.optimize.minimize(
                    lambda X: factor * self.gp.predict(X, n=self.n, return_cov=False)[0, 0],
                    scipy.mean(self.bounds, axis=1),
                    method='SLSQP',
                    bounds=self.bounds
                )
            except AttributeError:
                res = wrap_fmin_slsqp(
                    lambda X: factor * self.gp.predict(X, n=self.n, return_cov=False)[0, 0],
                    scipy.mean(self.bounds, axis=1),
                    opt_kwargs={'bounds': self.bounds, 'iprint': 0}
                )
                
            if not res.success:
                warnings.warn("Solver reports failure, extremum was likely NOT "
                              "found. Status: %d, Message: '%s'"
                              % (res.status, res.message),
                              RuntimeWarning)
            val = factor * res.fun
        if self.type_ == 'gt':
            return val - self.boundary_val
        else:
            return self.boundary_val - val
