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
from .utils import wrap_fmin_slsqp, univariate_envelope_plot, CombinedBounds, unique_rows, plot_sampler

import scipy
import scipy.linalg
import scipy.optimize
import scipy.stats
import numpy.random
import numpy.linalg
import sys
import warnings
import multiprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
try:
    import emcee
except ImportError:
    warnings.warn("Could not import emcee: MCMC sampling will not be available.")
try:
    import triangle
except ImportError:
    warnings.warn("Could not import triangle: plotting of MCMC results will not "
                  "be available.")

class GaussianProcess(object):
    r"""Gaussian process.
    
    If called with one argument, an untrained Gaussian process is constructed
    and data must be added with the :py:meth:`add_data` method. If called with
    the optional keywords, the values given are used as the data. It is always
    possible to add additional data with :py:meth:`add_data`.
    
    Note that the attributes have no write protection, but you should always
    add data with :py:meth:`add_data` to ensure internal consistency.
    
    Parameters
    ----------
    k : :py:class:`~gptools.kernel.core.Kernel` instance
        Kernel instance corresponding to the desired noise-free covariance
        kernel of the Gaussian process. The noise is handled separately either
        through specification of `err_y`, or in a separate kernel. This allows
        noise-free predictions when needed.
    noise_k : :py:class:`~gptools.kernel.core.Kernel` instance
        Kernel instance corresponding to the noise portion of the desired
        covariance kernel of the Gaussian process. Note that you DO NOT need to
        specify this if the extent of the noise you want to represent is
        contained in `err_y` (or if your data are noiseless). Default value is
        None, which results in the :py:class:`~gptools.kernel.noise.ZeroKernel`
        (noise specified elsewhere or not present).
    diag_factor : float, optional
        Factor of :py:attr:`sys.float_info.epsilon` which is added to the
        diagonal of the total `K` matrix to improve the stability of the
        Cholesky decomposition. If you are having issues, try increasing this by
        a factor of 10 at a time. Default is 1e2.
    mu : :py:class:`~gptools.mean.MeanFunction` instance
        The mean function of the Gaussian process. Default is None (zero mean
        prior).
    
    NOTE
        The following are all passed to :py:meth:`add_data`, refer to its
        docstring.
    
    X : array, (`M`, `D`), optional
        `M` input values of dimension `D`. Default value is None (no data).
    y : array, (`M`,), optional
        `M` data target values. Default value is None (no data).
    err_y : array, (`M`,), optional
        Error (given as standard deviation) in the `M` training target values.
        Default value is 0 (noiseless observations).
    n : array, (`M`, `D`) or scalar float, optional
        Non-negative integer values only. Degree of derivative for each target.
        If `n` is a scalar it is taken to be the value for all points in `y`.
        Otherwise, the length of n must equal the length of `y`. Default value
        is 0 (observation of target value). If non-integer values are passed,
        they will be silently rounded.
    T : array, (`M`, `N`), optional
        Linear transformation to get from latent variables to data in the
        argument `y`. When `T` is passed the argument `y` holds the transformed
        quantities `y=TY(X)` where `y` are the observed values of the
        transformed quantities, `T` is the transformation matrix and `Y(X)` is
        the underlying (untransformed) values of the function to be fit that
        enter into the transformation. When `T` is `M`-by-`N` and `y` has `M`
        elements, `X` and `n` will both be `N`-by-`D`. Default is None (no
        transformation).
    
    Attributes
    ----------
    k : :py:class:`~gptools.kernel.core.Kernel` instance
        The non-noise portion of the covariance kernel.
    noise_k : :py:class:`~gptools.kernel.core.Kernel` instance
        The noise portion of the covariance kernel.
    X : array, (`M`, `D`)
        The `M` training input values, each of which is of dimension `D`.
    y : array, (`M`,)
        The `M` training target values.
    err_y : array, (`M`,)
        The error in the `M` training input values.
    n : array, (`M`, `D`)
        The orders of derivatives that each of the `M` training points represent, indicating the order of derivative with respect to each of the `D` dimensions.
    T : array, (`M`, `N`)
        The transformation matrix applied to the data. If this is not None, `X` and `n` will be `N`-by-`D`.
    K_up_to_date : bool
        True if no data have been added since the last time the internal state was updated with a call to :py:meth:`compute_K_L_alpha_ll`.
    K : array, (`M`, `M`)
        Covariance matrix between all of the training inputs.
    noise_K : array, (`M`, `M`)
        Noise portion of the covariance matrix between all of the training inputs. Only includes the noise from :py:attr:`noise_k`, not from :py:attr:`err_y`.
    L : array, (`M`, `M`)
        Cholesky decomposition of the combined covariance matrix between all of the training inputs.
    alpha : array, (`M`, 1)
        Solution to :math:`K\alpha=y`.
    ll : float
        Log-likelihood of the data given the model.
    diag_factor : float
        The factor of :py:attr:`sys.float_info.epsilon` which is added to the diagonal of the `K` matrix to improve stability.
    mu : :py:class:`~gptools.mean.MeanFunction` instance
        The mean function.
        
    
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
    def __init__(self, k, noise_k=None, X=None, y=None, err_y=0, n=0, T=None,
                 diag_factor=1e2, mu=None):
        if not isinstance(k, Kernel):
            raise TypeError(
                "Argument k must be an instance of Kernel when constructing "
                "GaussianProcess!"
            )
        if noise_k is None:
            noise_k = ZeroKernel(k.num_dim)
        else:
            if not isinstance(noise_k, Kernel):
                raise TypeError(
                    "Keyword noise_k must be an instance of Kernel when "
                    "constructing GaussianProcess!"
                )
        
        self.mu = mu
        self.diag_factor = diag_factor
        self.k = k
        self.noise_k = noise_k
        self.y = scipy.array([], dtype=float)
        self.X = None
        self.err_y = scipy.array([], dtype=float)
        self.n = None
        self.T = None
        
        if X is not None:
            if y is None:
                raise GPArgumentError(
                    "Must pass both X and y when constructing GaussianProcess!"
                )
            else:
                self.add_data(X, y, err_y=err_y, n=n, T=T)
        elif X is None and y is not None:
            raise GPArgumentError(
                "Must pass both X and y when constructing GaussianProcess!"
            )
        else:
            self.K_up_to_date = False
    
    # The following are getters/setters for the (hyper)parameters of the model.
    # Right now they pull from the kernel, noise kernel and mean function.
    # Modify them to add more complicated things you want to infer.
    
    # TODO: These getters don't handle assignment by index!
    
    @property
    def hyperprior(self):
        """Combined hyperprior for the kernel, noise kernel and (if present) mean function.
        """
        hp = self.k.hyperprior * self.noise_k.hyperprior
        if self.mu is not None:
            hp *= self.mu.hyperprior
        return hp
    
    # TODO: Is there a clever way to globally set the hyperprior?
    # @hyperprior.setter
    # def hyperprior(self, value):
    #     pass
    
    @property
    def fixed_params(self):
        fp = CombinedBounds(self.k.fixed_params, self.noise_k.fixed_params)
        if self.mu is not None:
            fp = CombinedBounds(fp, self.mu.fixed_params)
        return fp
    
    @fixed_params.setter
    def fixed_params(self, value):
        value = scipy.asarray(value, dtype=bool)
        self.k.fixed_params = value[:self.k.num_params]
        self.noise_k.fixed_params = value[self.k.num_params:self.k.num_params + self.noise_k.num_params]
        if self.mu is not None:
            self.mu.fixed_params = value[self.k.num_params + self.noise_k.num_params:]
    
    @property
    def params(self):
        p = CombinedBounds(self.k.params, self.noise_k.params)
        if self.mu is not None:
            p = CombinedBounds(p, self.mu.params)
        return p
    
    @params.setter
    def params(self, value):
        value = scipy.asarray(value, dtype=float)
        self.K_up_to_date = False
        self.k.params = value[:self.k.num_params]
        self.noise_k.params = value[self.k.num_params:self.k.num_params + self.noise_k.num_params]
        if self.mu is not None:
            self.mu.params = value[self.k.num_params + self.noise_k.num_params:]
    
    @property
    def param_bounds(self):
        return self.hyperprior.bounds
    
    @param_bounds.setter
    def param_bounds(self, value):
        self.hyperprior.bounds = value
    
    @property
    def param_names(self):
        pn = CombinedBounds(self.k.param_names, self.noise_k.param_names)
        if self.mu is not None:
            pn = CombinedBounds(pn, self.mu.param_names)
        return pn
    
    @param_names.setter
    def param_names(self, value):
        self.k.param_names = value[:self.k.num_params]
        self.noise_k.param_names = value[self.k.num_params:self.k.num_params + self.noise_k.num_params]
        if self.mu is not None:
            self.mu.param_names = value[self.k.num_params + self.noise_k.num_params:]
    
    @property
    def free_params(self):
        p = CombinedBounds(self.k.free_params, self.noise_k.free_params)
        if self.mu is not None:
            p = CombinedBounds(p, self.mu.free_params)
        return p
    
    @free_params.setter
    def free_params(self, value):
        """Set the free parameters. Note that this bypasses enforce_bounds.
        """
        value = scipy.asarray(value, dtype=float)
        self.K_up_to_date = False
        self.k.free_params = value[:self.k.num_free_params]
        self.noise_k.free_params = value[self.k.num_free_params:self.k.num_free_params + self.noise_k.num_free_params]
        if self.mu is not None:
            self.mu.free_params = value[self.k.num_free_params + self.noise_k.num_free_params:]
    
    @property
    def free_param_bounds(self):
        fpb = CombinedBounds(self.k.free_param_bounds, self.noise_k.free_param_bounds)
        if self.mu is not None:
            fpb = CombinedBounds(fpb, self.mu.free_param_bounds)
        return fpb
    
    @free_param_bounds.setter
    def free_param_bounds(self, value):
        value = scipy.asarray(value, dtype=float)
        self.k.free_param_bounds = value[:self.k.num_free_params]
        self.noise_k.free_param_bounds = value[self.k.num_free_params:self.k.num_free_params + self.noise_k.num_free_params]
        if self.mu is not None:
            self.mu.free_param_bounds = value[self.k.num_free_params + self.noise_k.num_free_params:]
    
    @property
    def free_param_names(self):
        p = CombinedBounds(self.k.free_param_names, self.noise_k.free_param_names)
        if self.mu is not None:
            p = CombinedBounds(p, self.mu.free_param_names)
        return p
    
    @free_param_names.setter
    def free_param_names(self, value):
        value = scipy.asarray(value, dtype=str)
        self.K_up_to_date = False
        self.k.free_param_names = value[:self.k.num_free_params]
        self.noise_k.free_param_names = value[self.k.num_free_params:self.k.num_free_params + self.noise_k.num_free_params]
        if self.mu is not None:
            self.mu.free_param_names = value[self.k.num_free_params + self.noise_k.num_free_params:]
    
    def add_data(self, X, y, err_y=0, n=0, T=None):   
        """Add data to the training data set of the GaussianProcess instance.
        
        Parameters
        ----------
        X : array, (`M`, `D`)
            `M` input values of dimension `D`.
        y : array, (`M`,)
            `M` target values.
        err_y : array, (`M`,) or scalar float, optional
            Non-negative values only. Error given as standard deviation) in the
            `M` target values. If `err_y` is a scalar, the data set is taken to
            be homoscedastic (constant error). Otherwise, the length of `err_y`
            must equal the length of `y`. Default value is 0 (noiseless
            observations).
        n : array, (`M`, `D`) or scalar float, optional
            Non-negative integer values only. Degree of derivative for each
            target. If `n` is a scalar it is taken to be the value for all
            points in `y`. Otherwise, the length of n must equal the length of
            `y`. Default value is 0 (observation of target value). If
            non-integer values are passed, they will be silently rounded.
        T : array, (`M`, `N`), optional
            Linear transformation to get from latent variables to data in the
            argument `y`. When `T` is passed the argument `y` holds the
            transformed quantities `y=TY(X)` where `y` are the observed values
            of the transformed quantities, `T` is the transformation matrix and
            `Y(X)` is the underlying (untransformed) values of the function to
            be fit that enter into the transformation. When `T` is `M`-by-`N`
            and `y` has `M` elements, `X` and `n` will both be `N`-by-`D`.
            Default is None (no transformation).
        
        Raises
        ------
        ValueError
            Bad shapes for any of the inputs, negative values for `err_y` or `n`.
        """
        # Verify y has only one non-trivial dimension:
        try:
            iter(y)
        except TypeError:
            y = scipy.asarray([y], dtype=float)
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
        
        # Handle scalar training input or convert array input into 2d.
        try:
            iter(X)
        except TypeError:
            X = [X]
        X = scipy.atleast_2d(scipy.asarray(X, dtype=float))
        # Correct single-dimension inputs:
        if self.num_dim == 1 and X.shape[0] == 1:
            X = X.T
        if T is None and X.shape != (len(y), self.num_dim):
            raise ValueError("Shape of training inputs must be (len(y), "
                             "k.num_dim)! X given has shape %s, shape of y is "
                             "%s and num_dim=%d."
                             % (X.shape, y.shape, self.num_dim))
        
        # Handle scalar derivative orders or verify shape of array derivative
        # orders matches shape of y:
        try:
            iter(n)
        except TypeError:
            n = n * scipy.ones_like(X, dtype=int)
        else:
            n = scipy.atleast_2d(scipy.asarray(n, dtype=int))
            # Correct single-dimension inputs:
            if self.num_dim == 1 and n.shape[1] != 1:
                n = n.T
            if n.shape != X.shape:
                raise ValueError("When using array-like n, shape must be "
                                 "(len(y), k.num_dim)! Shape of n given is %s, "
                                 "shape of y given is %s and num_dim=%d."
                                 % (n.shape, y.shape, self.num_dim))
        if (n < 0).any():
            raise ValueError("All elements of n must be non-negative integers!")
        
        # Handle transform:
        if T is None and self.T is not None:
            T = scipy.eye(len(y))
        if T is not None:
            T = scipy.atleast_2d(scipy.asarray(T, dtype=float))
            if T.ndim != 2:
                raise ValueError("T must have exactly 2 dimensions!")
            if T.shape[0] != len(y):
                raise ValueError(
                    "T must have as many rows are there are elements in y!"
                )
            if T.shape[1] != X.shape[0]:
                raise ValueError(
                    "There must be as many columns in T as there are rows in X!"
                )
            if self.T is None and self.X is not None:
                self.T = scipy.eye(len(self.y))
            
            if self.T is None:
                self.T = T
            else:
                self.T = scipy.linalg.block_diag(self.T, T)
        
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
    
    def condense_duplicates(self):
        """Condense duplicate points using a transformation matrix.
        
        This is useful if you have multiple non-transformed points at the same
        location or multiple transformed points that use the same quadrature
        points.
        
        Won't change the GP if all of the rows of [X, n] are unique. Will create
        a transformation matrix T if necessary. Note that the order of the
        points in [X, n] will be arbitrary after this operation.
        """
        unique, inv = unique_rows(
            scipy.hstack((self.X, self.n)),
            return_inverse=True
        )
        # Only proceed if there is anything to be gained:
        if len(unique) != len(self.X):
            if self.T is None:
                self.T = scipy.eye(len(self.y))
            new_T = scipy.zeros((len(self.y), unique.shape[0]))
            for j in xrange(0, len(inv)):
                new_T[:, inv[j]] += self.T[:, j]
            self.T = new_T
            self.n = unique[:, self.X.shape[1]:]
            self.X = unique[:, :self.X.shape[1]]
    
    def remove_outliers(self, thresh=3, **predict_kwargs):
        """Remove outliers from the GP.
        
        Removes points that are more than `thresh` * `err_y` away from the GP
        mean. Note that this is only very rough in that it ignores the
        uncertainty in the GP mean at any given point. But you should only be
        using this as a rough way of removing bad channels, anyways!
        
        Returns the values that were removed and a boolean array indicating
        where the removed points were.
        
        Parameters
        ----------
        thresh : float, optional
            The threshold as a multiplier times `err_y`. Default is 3 (i.e.,
            throw away all 3-sigma points).
        **predict_kwargs : optional kwargs
            All additional kwargs are passed to :py:meth:`predict`. You can, for
            instance, use this to make it use MCMC to evaluate the mean. (If you
            don't use MCMC, then the current value of the hyperparameters is
            used.)
        
        Returns
        -------
        X_bad : array
            Input values of the bad points.
        y_bad : array
            Bad values.
        err_y_bad : array
            Uncertainties on the bad values.
        n_bad : array
            Derivative order of the bad values.
        bad_idxs : array
            Array of booleans with the original shape of X with True wherever
            a point was taken to be bad and subsequently removed.
        T_bad : array
            Transformation matrix of returned points. Only returned if
            :py:attr:`T` is not None for the instance.
        """
        # Find where a point lies more than thresh*err_y away from the mean:
        # This is naive as it does not account for the posterior variance in the
        # GP itself, but should work as a first-cut approach to deleting
        # outliers.
        mean = self.predict(
            self.X, n=self.n, noise=False, return_std=False,
            output_transform=self.T, **predict_kwargs
        )
        deltas = scipy.absolute(mean - self.y) / self.err_y
        deltas[self.err_y == 0] = 0
        bad_idxs = (deltas >= thresh)
        good_idxs = ~bad_idxs
        
        # Pull out the old values so they can be returned:
        y_bad = self.y[bad_idxs]
        err_y_bad = self.err_y[bad_idxs]
        if self.T is not None:
            T_bad = self.T[bad_idxs, :]
            non_zero_cols = (T_bad != 0).all(axis=0)
            T_bad = T_bad[:, non_zero_cols]
            X_bad = self.X[non_zero_cols, :]
            n_bad = self.n[non_zero_cols, :]
        else:
            X_bad = self.X[bad_idxs, :]
            n_bad = self.n[bad_idxs, :]
        
        # Delete the offending points:
        if self.T is None:
            self.X = self.X[good_idxs, :]
            self.n = self.n[good_idxs, :]
        else:
            self.T = self.T[good_idxs, :]
            non_zero_cols = (self.T != 0).all(axis=0)
            self.T = self.T[:, non_zero_cols]
            self.X = self.X[non_zero_cols, :]
            self.n = self.n[non_zero_cols, :]
        self.y = self.y[good_idxs]
        self.err_y = self.err_y[good_idxs]
        self.K_up_to_date = False
        
        if self.T is None:
            return (X_bad, y_bad, err_y_bad, n_bad, bad_idxs)
        else:
            return (X_bad, y_bad, err_y_bad, n_bad, bad_idxs, T_bad)
    
    def optimize_hyperparameters(self, method='SLSQP', opt_kwargs={},
                                 verbose=False, random_starts=None, num_proc=None):
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
            :py:func:`scipy.optimize.minimize`. Refer to that function's
            docstring for valid options. Note that if you use `jac` = True (i.e.,
            optimization function returns Jacobian) you should also set `args`
            = (True,) to tell :py:meth:`update_hyperparameters` to compute and
            return the Jacobian. Default is: {}.
        verbose : bool, optional
            Whether or not the output should be verbose. If True, the entire
            :py:class:`Result` object from :py:func:`scipy.optimize.minimize` is
            printed. If False, status information is only printed if the
            `success` flag from :py:func:`minimize` is False. Default is False.
        random_starts : non-negative int, optional
            Number of times to randomly perturb the starting guesses
            (distributed uniformly within their bounds) in order to seek the
            global minimum. If None, then `num_proc` random starts will be
            performed. Default is None (do number of random starts equal to the
            number of processors allocated). Note that for `random_starts` != 0,
            the initial guesses provided are not actually used.
        num_proc : non-negative int or None
            Number of processors to use with random starts. If 0, processing is
            not done in parallel. If None, all available processors are used.
            Default is None (use all available processors).
        """
        if opt_kwargs is None:
            opt_kwargs = {}
        else:
            opt_kwargs = dict(opt_kwargs)
        if 'method' in opt_kwargs:
            warnings.warn(
                "Key 'method' is present in opt_kwargs, will override option "
                "specified with method kwarg.",
                RuntimeWarning
            )
        else:
            opt_kwargs['method'] = method
        
        if num_proc is None:
            num_proc = multiprocessing.cpu_count()
        
        param_ranges = scipy.asarray(self.free_param_bounds, dtype=float)
        # Replace unbounded variables with something big:
        param_ranges[scipy.where(scipy.isnan(param_ranges[:, 0])), 0] = -1e16
        param_ranges[scipy.where(scipy.isnan(param_ranges[:, 1])), 1] = 1e16
        if random_starts == 0:
            num_proc = 0
            param_samples = [self.free_params]
        else:
            if random_starts is None:
                random_starts = max(num_proc, 1)
            # Distribute random guesses according to the hyperprior:
            param_samples = self.hyperprior.random_draw(size=random_starts).T
            param_samples = param_samples[:, ~self.fixed_params]
        if 'bounds' not in opt_kwargs:
            opt_kwargs['bounds'] = param_ranges
        if num_proc <= 1:
            res = []
            for samp in param_samples:
                try:
                    res += [
                        scipy.optimize.minimize(
                            self.update_hyperparameters,
                            samp,
                            **opt_kwargs
                        )
                    ]
                except AttributeError:
                    warnings.warn(
                        "scipy.optimize.minimize not available, defaulting to "
                        "fmin_slsqp.",
                        RuntimeWarning
                    )
                    res += [
                        wrap_fmin_slsqp(
                            self.update_hyperparameters,
                            samp,
                            opt_kwargs=opt_kwargs
                        )
                    ]
                except:
                    warnings.warn(
                        "Minimizer failed, skipping sample. Error is: %s: %s. "
                        "State of params is: %s %s"
                        % (
                            sys.exc_info()[0],
                            sys.exc_info()[1],
                            str(self.k.free_params),
                            str(self.noise_k.free_params)
                        ),
                        RuntimeWarning
                    )
        else:
            pool = multiprocessing.Pool(processes=num_proc)
            try:
                res = pool.map(
                    _OptimizeHyperparametersEval(self, opt_kwargs),
                    param_samples
                )
            finally:
                pool.close()
            # Filter out the failed convergences:
            res = [r for r in res if r is not None]
        try:
            res_min = min(res, key=lambda r: r.fun)
        except ValueError:
            raise ValueError(
                "Optimizer failed to find a valid solution. Try changing the "
                "parameter bounds, picking a new initial guess or increasing the "
                "number of random starts."
            )
        
        self.update_hyperparameters(res_min.x)
        if verbose:
            print("Got %d completed starts, optimal result is:" % (len(res),))
            print(res_min)
            print("\nLL\t%.3g" % (-1 * res_min.fun))
            for v, l in zip(res_min.x, self.free_param_names):
                print("%s\t%.3g" % (l.translate(None, '\\'), v))
        if not res_min.success:
            warnings.warn(
                "Optimizer %s reports failure, selected hyperparameters are "
                "likely NOT optimal. Status: %d, Message: '%s'. Try adjusting "
                "bounds, initial guesses or the number of random starts used."
                % (
                    method,
                    res_min.status,
                    res_min.message
                ),
                RuntimeWarning
            )
        bounds = scipy.asarray(self.free_param_bounds)
        # Augment the bounds a little bit to catch things that are one step away:
        if ((res_min.x <= 1.001 * bounds[:, 0]).any() or
            (res_min.x >= 0.999 * bounds[:, 1]).any()):
            warnings.warn(
                "Optimizer appears to have hit/exceeded the bounds. Bounds are:\n"
                "%s\n, solution is:\n%s. Try adjusting bounds, initial guesses "
                "or the number of random starts used."
                % (str(bounds), str(res_min.x),)
            )
        return (res_min, len(res))
    
    def predict(self, Xstar, n=0, noise=False, return_std=True, return_cov=False,
                full_output=False, return_samples=False, num_samples=1,
                samp_kwargs={}, use_MCMC=False, full_MC=False, rejection_func=None,
                ddof=1, output_transform=None, **kwargs):
        """Predict the mean and covariance at the inputs `Xstar`.
        
        The order of the derivative is given by `n`. The keyword `noise` sets
        whether or not noise is included in the prediction.
        
        Parameters
        ----------
        Xstar : array, (`M`, `D`)
            `M` test input values of dimension `D`.
        n : array, (`M`, `D`) or scalar, non-negative int, optional
            Order of derivative to predict (0 is the base quantity). If `n` is
            scalar, the value is used for all points in `Xstar`. If non-integer
            values are passed, they will be silently rounded. Default is 0
            (return base quantity).
        noise : bool, optional
            Whether or not noise should be included in the covariance. Default
            is False (no noise in covariance).
        return_std : bool, optional
            Set to True to compute and return the standard deviation for the
            predictions, False to skip this step. Default is True (return tuple
            of (`mean`, `std`)).
        return_cov : bool, optional
            Set to True to compute and return the full covariance matrix for the
            predictions. This overrides the `return_std` keyword. If you want
            both the standard deviation and covariance matrix pre-computed, use
            the `full_output` keyword.
        full_output : bool, optional
            Set to True to return the full outputs in a dictionary with keys:
            
                ==== ==========================================================================
                mean mean of GP at requested points
                std  standard deviation of GP at requested points
                cov  covariance matrix for values of GP at requested points
                samp random samples of GP at requested points (only if `return_sample` is True)
                ==== ==========================================================================
        
        return_samples : bool, optional
            Set to True to compute and return samples of the GP in addition to
            computing the mean. Only done if `full_output` is True. Default is
            False.
        num_samples : int, optional
            Number of samples to compute. If using MCMC this is the number of
            samples per MCMC sample, if using present values of hyperparameters
            this is the number of samples actually returned. Default is 1.
        samp_kwargs : dict, optional
            Additional keywords to pass to :py:meth:`draw_sample` if
            `return_samples` is True. Default is {}.
        use_MCMC : bool, optional
            Set to True to use :py:meth:`predict_MCMC` to evaluate the prediction
            marginalized over the hyperparameters.
        full_MC : bool, optional
            Set to True to compute the mean and covariance matrix using Monte
            Carlo sampling of the posterior. The samples will also be returned
            if full_output is True. Default is False (don't use full sampling).
        rejection_func : callable, optional
            Any samples where this function evaluates False will be rejected,
            where it evaluates True they will be kept. Default is None (no
            rejection). Only has an effect if `full_MC` is True.
        ddof : int, optional
            The degree of freedom correction to use when computing the covariance
            matrix when `full_MC` is True. Default is 1 (unbiased estimator).
        output_transform: array, (`L`, `M`), optional
            Matrix to use to transform the output vector of length `M` to one of
            length `L`. This can, for instance, be used to compute integrals.
        **kwargs : optional kwargs
            All additional kwargs are passed to :py:meth:`predict_MCMC` if
            `use_MCMC` is True.
        
        Returns
        -------
        mean : array, (`M`,)
            Predicted GP mean. Only returned if `full_output` is False.
        std : array, (`M`,)
            Predicted standard deviation, only returned if `return_std` is True, `return_cov` is False and `full_output` is False.
        cov : array, (`M`, `M`)
            Predicted covariance matrix, only returned if `return_cov` is True and `full_output` is False.
        full_output : dict
            Dictionary with fields for mean, std, cov and possibly random samples. Only returned if `full_output` is True.
        
        Raises
        ------
        ValueError
            If `n` is not consistent with the shape of `Xstar` or is not entirely
            composed of non-negative integers.
        """
        if use_MCMC:
            res = self.predict_MCMC(
                Xstar,
                n=n,
                noise=noise,
                return_std=return_std or full_output,
                return_cov=return_cov or full_output,
                return_samples=full_output and (return_samples or rejection_func),
                num_samples=num_samples,
                samp_kwargs=samp_kwargs,
                full_MC=full_MC,
                rejection_func=rejection_func,
                ddof=ddof,
                output_transform=output_transform,
                **kwargs
            )
            if full_output:
                return res
            elif return_cov:
                return (res['mean'], res['cov'])
            elif return_std:
                return (res['mean'], res['std'])
            else:
                return res['mean']
        else:
            # Process Xstar:
            Xstar = scipy.atleast_2d(scipy.asarray(Xstar, dtype=float))
            # Handle 1d x case where array is passed in:
            if self.num_dim == 1 and Xstar.shape[0] == 1:
                Xstar = Xstar.T
            if Xstar.shape[1] != self.num_dim:
                raise ValueError(
                    "Second dimension of Xstar must be equal to self.num_dim! "
                    "Shape of Xstar given is %s, num_dim is %d."
                    % (Xstar.shape, self.num_dim)
                )
            
            # Process T:
            if output_transform is not None:
                output_transform = scipy.atleast_2d(scipy.asarray(output_transform, dtype=float))
                if output_transform.ndim != 2:
                    raise ValueError(
                        "output_transform must have exactly 2 dimensions! Shape "
                        "of output_transform given is %s."
                        % (output_transform.shape,)
                    )
                if output_transform.shape[1] != Xstar.shape[0]:
                    raise ValueError(
                        "output_transform must have the same number of columns "
                        "the number of rows in Xstar! Shape of output_transform "
                        "given is %s, shape of Xstar is %s."
                        % (output_transform.shape, Xstar.shape,)
                    )
            
            # Process n:
            try:
                iter(n)
            except TypeError:
                n = n * scipy.ones(Xstar.shape, dtype=int)
            else:
                n = scipy.atleast_2d(scipy.asarray(n, dtype=int))
                if self.num_dim == 1 and n.shape[0] == 1:
                    n = n.T
                if n.shape != Xstar.shape:
                    raise ValueError(
                        "When using array-like n, shape must match shape of Xstar! "
                        "Shape of n given is %s, shape of Xstar given is %s."
                        % (n.shape, Xstar.shape)
                    )
            if (n < 0).any():
                raise ValueError("All elements of n must be non-negative integers!")
            
            self.compute_K_L_alpha_ll()
            Kstar = self.compute_Kij(self.X, Xstar, self.n, n)
            if noise:
                Kstar = Kstar + self.compute_Kij(self.X, Xstar, self.n, n, noise=True)
            if self.T is not None:
                Kstar = self.T.dot(Kstar)
            mean = Kstar.T.dot(self.alpha)
            if self.mu is not None:
                mean += scipy.atleast_2d(self.mu(Xstar, n)).T
            if output_transform is not None:
                mean = output_transform.dot(mean)
            mean = mean.ravel()
            if return_std or return_cov or full_output or full_MC:
                v = scipy.linalg.solve_triangular(self.L, Kstar, lower=True)
                Kstarstar = self.compute_Kij(Xstar, None, n, None)
                if noise:
                    Kstarstar = Kstarstar + self.compute_Kij(Xstar, None, n, None, noise=True)
                covariance = Kstarstar - v.T.dot(v)
                if output_transform is not None:
                    covariance = output_transform.dot(covariance.dot(output_transform.T))
                if return_samples or full_MC:
                    samps = self.draw_sample(
                        Xstar, n=n, num_samp=num_samples, mean=mean,
                        cov=covariance, **samp_kwargs
                    )
                    if rejection_func:
                        good_samps = []
                        for samp in samps.T:
                            if rejection_func(samp):
                                good_samps.append(samp)
                        if len(good_samps) == 0:
                            raise ValueError("Did not get any good samples!")
                        samps = scipy.asarray(good_samps, dtype=float).T
                    if full_MC:
                        mean = scipy.mean(samps, axis=1)
                        covariance = scipy.cov(samps, rowvar=1, ddof=ddof)
                std = scipy.sqrt(scipy.diagonal(covariance))
                if full_output:
                    out = {
                        'mean': mean,
                        'std': std,
                        'cov': covariance
                    }
                    if return_samples or full_MC:
                        out['samp'] = samps
                    return out
                else:
                    if return_cov:
                        return (mean, covariance)
                    elif return_std:
                        return (mean, std)
                    else:
                        return mean
            else:
                return mean
    
    def plot(self, X=None, n=0, ax=None, envelopes=[1, 3], base_alpha=0.375,
             return_prediction=False, return_std=True, full_output=False,
             plot_kwargs={}, **kwargs):
        """Plots the Gaussian process using the current hyperparameters. Only for num_dim <= 2.
        
        Parameters
        ----------
        X : array-like (`M`,) or (`M`, `num_dim`), optional
            The values to evaluate the Gaussian process at. If None, then 100
            points between the minimum and maximum of the data's X are used.
            Default is None (use 100 points between min and max).
        n : int or list, optional
            The order of derivative to compute. For num_dim=1, this must be an
            int. For num_dim=2, this must be a list of ints of length 2.
            Default is 0 (don't take derivative).
        ax : axis instance, optional
            Axis to plot the result on. If no axis is passed, one is created.
            If the string 'gca' is passed, the current axis (from plt.gca())
            is used. If X_dim = 2, the axis must be 3d.
        envelopes: list of float, optional
            +/-n*sigma envelopes to plot. Default is [1, 3].
        base_alpha : float, optional
            Alpha value to use for +/-1*sigma envelope. All other envelopes env
            are drawn with base_alpha/env. Default is 0.375.
        return_prediction : bool, optional
            If True, the predicted values are also returned. Default is False.
        return_std : bool, optional
            If True, the standard deviation is computed and returned along with
            the mean when `return_prediction` is True. Default is True.
        full_output : bool, optional
            Set to True to return the full outputs in a dictionary with keys:
            
                ==== ==========================================================================
                mean mean of GP at requested points
                std  standard deviation of GP at requested points
                cov  covariance matrix for values of GP at requested points
                samp random samples of GP at requested points (only if `return_sample` is True)
                ==== ==========================================================================
            
        plot_kwargs : dict, optional
            The entries in this dictionary are passed as kwargs to the plotting
            command used to plot the mean. Use this to, for instance, change the
            color, line width and line style.
        **kwargs : extra arguments for predict, optional
            Extra arguments that are passed to :py:meth:`predict`.
        
        Returns
        -------
        ax : axis instance
            The axis instance used.
        mean : :py:class:`Array`, (`M`,)
            Predicted GP mean. Only returned if `return_prediction` is True and `full_output` is False.
        std : :py:class:`Array`, (`M`,)
            Predicted standard deviation, only returned if `return_prediction` and `return_std` are True and `full_output` is False.
        full_output : dict
            Dictionary with fields for mean, std, cov and possibly random samples. Only returned if `return_prediction` and `full_output` are True.
        """
        if self.num_dim > 2:
            raise ValueError("Plotting is not supported for num_dim > 2!")
        
        if self.num_dim == 1:
            if X is None:
                X = scipy.linspace(self.X.min(), self.X.max(), 100)
        elif self.num_dim == 2:
            if X is None:
                x1 = scipy.linspace(self.X[:, 0].min(), self.X[:, 0].max(), 50)
                x2 = scipy.linspace(self.X[:, 1].min(), self.X[:, 1].max(), 50)
                X1, X2 = scipy.meshgrid(x1, x2)
                X1 = X1.flatten()
                X2 = X2.flatten()
                X = scipy.hstack((scipy.atleast_2d(X1).T, scipy.atleast_2d(X2).T))
            else:
                X1 = scipy.asarray(X[:, 0]).flatten()
                X2 = scipy.asarray(X[:, 1]).flatten()
        
        if envelopes or (return_prediction and (return_std or full_output)):
            out = self.predict(X, n=n, full_output=True, **kwargs)
            mean = out['mean']
            std = out['std']
        else:
            mean = self.predict(X, n=n, return_std=False, **kwargs)
            std = None
        
        if self.num_dim == 1:
            univariate_envelope_plot(
                X,
                mean,
                std,
                ax=ax,
                base_alpha=base_alpha,
                envelopes=envelopes,
                **plot_kwargs
            )
        elif self.num_dim == 2:
            if ax is None:
                f = plt.figure()
                ax = f.add_subplot(111, projection='3d')
            elif ax == 'gca':
                ax = plt.gca()
            if 'linewidths' not in kwargs:
                kwargs['linewidths'] = 0
            s = ax.plot_trisurf(X1, X2, mean, **plot_kwargs)
            for i in envelopes:
                kwargs.pop('alpha', base_alpha)
                ax.plot_trisurf(X1, X2, mean - std, alpha=base_alpha / i, **kwargs)
                ax.plot_trisurf(X1, X2, mean + std, alpha=base_alpha / i, **kwargs)
        
        if return_prediction:
            if full_output:
                return (ax, out)
            elif return_std:
                return (ax, out['mean'], out['std'])
            else:
                return (ax, out['mean'])
        else:
            return ax
    
    def draw_sample(self, Xstar, n=0, num_samp=1, rand_vars=None,
                    rand_type='standard normal', diag_factor=1e3,
                    method='cholesky', num_eig=None, mean=None, cov=None,
                    **kwargs):
        """Draw a sample evaluated at the given points `Xstar`.
        
        Parameters
        ----------
        Xstar : array, (`M`, `D`)
            `M` test input values of dimension `D`.
        n : array, (`M`, `D`) or scalar, non-negative int, optional
            Derivative order to evaluate at. Default is 0 (evaluate value).
        noise : bool, optional
            Whether or not to include the noise components of the kernel in the
            sample. Default is False (no noise in samples).
        num_samp : Positive int, optional
            Number of samples to draw. Default is 1. Cannot be used in
            conjunction with `rand_vars`: If you pass both `num_samp` and
            `rand_vars`, `num_samp` will be silently ignored.
        rand_vars : array, (`M`, `P`), optional
            Vector of random variables :math:`u` to use in constructing the
            sample :math:`y_* = f_* + Lu`, where :math:`K=LL^T`. If None,
            values will be produced using
            :py:func:`numpy.random.multivariate_normal`. This allows you to use
            pseudo/quasi random numbers generated by an external routine.
            Default is None (use :py:func:`multivariate_normal` directly).
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
            If you are getting errors from :py:func:`scipy.linalg.cholesky`, try
            increasing this an order of magnitude at a time. This parameter only
            has an effect when using rand_vars. Default value is 1e3. 
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
        mean : array, (`M`,)
            If you have pre-computed the mean and covariance matrix, then you
            can simply pass them in with the `mean` and `cov` keywords to save
            on having to call :py:meth:`predict`.
        cov : array, (`M`, `M`)
            If you have pre-computed the mean and covariance matrix, then you
            can simply pass them in with the `mean` and `cov` keywords to save
            on having to call :py:meth:`predict`.
        **kwargs : optional kwargs
            All extra keyword arguments are passed to :py:meth:`predict` when
            evaluating the mean and covariance matrix of the GP.
        
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
        if mean is None or cov is None:
            out = self.predict(Xstar, n=n, full_output=True, **kwargs)
            mean = out['mean']
            cov = out['cov']
        if rand_vars is None and method != 'eig':
            try:
                return numpy.random.multivariate_normal(mean, cov, num_samp).T
            except numpy.linalg.LinAlgError as e:
                warnings.warn(
                    "Failure when drawing from MVN! Falling back on eig. "
                    "Exception was:\n%s"
                    % (e,),
                    RuntimeWarning
                )
                method = 'eig'
        
        if num_eig is None or num_eig > len(mean):
            num_eig = len(mean)
        elif num_eig < 1:
            num_eig = 1
        if rand_vars is None:
            rand_vars = numpy.random.standard_normal((num_eig, num_samp))
        valid_types = ('standard normal', 'uniform')
        if rand_type not in valid_types:
            raise ValueError("rand_type %s not recognized! Valid options "
                             "are: %s." % (rand_type, valid_types,))
        if rand_type == 'uniform':
            rand_vars = scipy.stats.norm.ppf(rand_vars)
        
        if method == 'cholesky':
            L = scipy.linalg.cholesky(
                cov + diag_factor * sys.float_info.epsilon * scipy.eye(cov.shape[0]),
                lower=True,
                check_finite=False
            )
        elif method == 'eig':
            # TODO: Add support for specifying cutoff eigenvalue!
            # Not technically lower triangular, but we'll keep the name L:
            eig, Q = scipy.linalg.eigh(
                cov + diag_factor * sys.float_info.epsilon * scipy.eye(cov.shape[0]),
                eigvals=(len(mean) - 1 - (num_eig - 1), len(mean) - 1)
            )
            Lam_1_2 = scipy.diag(scipy.sqrt(eig))
            L = Q.dot(Lam_1_2)
        else:
            raise ValueError("method %s not recognized!" % (method,))
        return scipy.atleast_2d(mean).T + L.dot(rand_vars[:num_eig, :])
    
    def update_hyperparameters(self, new_params, exit_on_bounds=True, inf_on_error=True):
        """Update the kernel's hyperparameters to the new parameters.
        
        This will call :py:meth:`compute_K_L_alpha_ll` to update the state
        accordingly.
        
        Parameters
        ----------
        new_params : :py:class:`Array` or other Array-like, length dictated by kernel
            New parameters to use.
        exit_on_bounds : bool, optional
            If True, the method will automatically exit if the hyperparameters
            are impossible given the hyperprior, without trying to update the
            internal state. This is useful during MCMC sampling and optimization.
            Default is True (don't perform update for impossible hyperparameters).
        inf_on_error : bool, optional
            If True, the method will return `scipy.inf` if the hyperparameters
            produce a linear algebra error upon trying to update the Gaussian
            process. Default is True (catch errors and return infinity).
        
        Returns
        -------
        -1*ll : float
            The updated log likelihood.
        """
        self.k.set_hyperparams(new_params[:len(self.k.free_params)])
        self.noise_k.set_hyperparams(
            new_params[len(self.k.free_params):len(self.k.free_params) + len(self.noise_k.free_params)]
        )
        if self.mu is not None:
            self.mu.set_hyperparams(
                new_params[len(self.k.free_params) + len(self.noise_k.free_params):]
            )
        self.K_up_to_date = False
        if exit_on_bounds:
            if scipy.isinf(self.hyperprior(self.params)):
                return scipy.inf
        try:
            self.compute_K_L_alpha_ll()
        except numpy.linalg.LinAlgError as e:
            if inf_on_error:
                warnings.warn(
                    "Failure when updating GP! Exception was:\n%s\n"
                    "State of params is: %s"
                    % (
                        e,
                        str(self.free_params[:]),
                    ),
                    RuntimeWarning
                )
                return scipy.inf
            else:
                raise e
        except Exception as e:
            if inf_on_error:
                warnings.warn(
                    "Unhandled exception! Exception was:\n%s\n"
                    "State of params is: %s"
                    % (
                    e,
                    str(self.free_params[:]))
                )
                return scipy.inf
            else:
                raise e
        return -1 * self.ll
    
    def compute_K_L_alpha_ll(self):
        r"""Compute `K`, `L`, `alpha` and log-likelihood according to the first part of Algorithm 2.1 in R&W.
        
        Computes `K` and the noise portion of `K` using :py:meth:`compute_Kij`,
        computes `L` using :py:func:`scipy.linalg.cholesky`, then computes
        `alpha` as `L.T\\(L\\y)`.
        
        Only does the computation if :py:attr:`K_up_to_date` is False --
        otherwise leaves the existing values.
        """
        if not self.K_up_to_date:
            y = self.y
            err_y = self.err_y
            self.K = self.compute_Kij(self.X, None, self.n, None, noise=False)
            self.noise_K = self.compute_Kij(self.X, None, self.n, None, noise=True)
            K = self.K
            noise_K = self.noise_K
            if self.T is not None:
                K = self.T.dot(K).dot(self.T.T)
                noise_K = self.T.dot(noise_K).dot(self.T.T)
            K_tot = (
                K +
                scipy.diag(err_y**2.0) +
                noise_K +
                self.diag_factor * sys.float_info.epsilon * scipy.eye(len(y))
            )
            self.L = scipy.linalg.cholesky(K_tot, lower=True)
            # Need to make the mean-subtracted y that appears in the expression
            # for alpha:
            if self.mu is not None:
                mu_alph = self.mu(self.X, self.n)
                if self.T is not None:
                    mu_alph = self.T.dot(mu_alph)
                y_alph = self.y - mu_alph
            else:
                y_alph = self.y
            self.alpha = scipy.linalg.solve_triangular(
                self.L.T,
                scipy.linalg.solve_triangular(
                    self.L,
                    scipy.atleast_2d(y_alph).T,
                    lower=True
                ),
                lower=False
            )
            self.ll = (
                -0.5 * scipy.atleast_2d(y_alph).dot(self.alpha) -
                scipy.log(scipy.diag(self.L)).sum() - 
                0.5 * len(y) * scipy.log(2.0 * scipy.pi)
            )[0, 0]
            # Apply hyperpriors:
            self.ll += self.hyperprior(self.params)
            self.K_up_to_date = True
    
    @property
    def num_dim(self):
        """The number of dimensions of the input data.
        
        Returns
        -------
        num_dim: int
            The number of dimensions of the input data as defined in the kernel.
        """
        return self.k.num_dim
    
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
        Xi : array, (`M`, `D`)
            `M` input values of dimension `D`.
        Xj : array, (`P`, `D`)
            `P` input values of dimension `D`.
        ni : array, (`M`,), non-negative integers
            `M` derivative orders with respect to the `Xi` coordinates.
        nj : array, (`P`,), non-negative integers
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
        Kij : array, (`M`, `P`)
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
        Kij = k(
            Xi_tile,
            Xj_tile,
            ni_tile,
            nj_tile,
            hyper_deriv=hyper_deriv,
            symmetric=symmetric
        )
        Kij = scipy.reshape(Kij, (Xi.shape[0], -1))
        
        return Kij
    
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
        present_free_params = scipy.concatenate(
            (self.k.free_params, self.noise_k.free_params)
        )
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
                raise ValueError("Length of num_pts must match the number of "
                                 "free parameters of kernel!")
        
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
            return -1.0 * self.update_hyperparameters(
                scipy.asarray(param_vals, dtype=float)
            )
        else:
            # Recursive case: call _compute_ll_matrix for each entry in param_vals[idx]:
            vals = scipy.zeros(num_pts[idx:], dtype=float)
            for k in xrange(0, len(param_vals[idx])):
                specific_param_vals = list(param_vals)
                specific_param_vals[idx] = param_vals[idx][k]
                vals[k] = self._compute_ll_matrix(
                    idx + 1,
                    specific_param_vals,
                    num_pts
                )
            return vals
    
    def sample_hyperparameter_posterior(self, nwalkers=200, nsamp=500, burn=0,
                                        thin=1, num_proc=None, sampler=None,
                                        plot_posterior=False,
                                        plot_chains=False, sampler_type='ensemble',
                                        ntemps=20, sampler_a=2.0):
        """Produce samples from the posterior for the hyperparameters using MCMC.
        
        Returns the sampler created, because storing it stops the GP from being
        pickleable. To add more samples to a previous sampler, pass the sampler
        instance in the `sampler` keyword.
        
        Parameters
        ----------
        nwalkers : int, optional
            The number of walkers to use in the sampler. Should be on the order
            of several hundred. Default is 200.
        nsamp : int, optional
            Number of samples (per walker) to take. Default is 500.
        burn : int, optional
            This keyword only has an effect on the corner plot produced when
            `plot_posterior` is True and the flattened chain plot produced
            when `plot_chains` is True. To perform computations with burn-in,
            see :py:meth:`compute_from_MCMC`. The number of samples to discard
            at the beginning of the chain. Default is 0.
        thin : int, optional
            This keyword only has an effect on the corner plot produced when
            `plot_posterior` is True and the flattened chain plot produced
            when `plot_chains` is True. To perform computations with thinning,
            see :py:meth:`compute_from_MCMC`. Every `thin`-th sample is kept.
            Default is 1.
        num_proc : int or None, optional
            Number of processors to use. If None, all available processors are
            used. Default is None (use all available processors).
        sampler : :py:class:`Sampler` instance
            The sampler to use. If the sampler already has samples, the most
            recent sample will be used as the starting point. Otherwise a
            random sample from the hyperprior will be used.
        plot_posterior : bool, optional
            If True, a corner plot of the posterior for the hyperparameters
            will be generated. Default is False.
        plot_chains : bool, optional
            If True, a plot showing the history and autocorrelation of the
            chains will be produced.
        sampler_type : str, optional
            The type of sampler to use. Valid options are "ensemble" (affine-
            invariant ensemble sampler) and "pt" (parallel-tempered ensemble
            sampler).
        ntemps : int, optional
            Number of temperatures to use with the parallel-tempered ensemble
            sampler.
        sampler_a : float, optional
            Scale of the proposal distribution.
        """
        if num_proc is None:
            num_proc = multiprocessing.cpu_count()
        # Needed for emcee to do it right:
        if num_proc == 0:
            num_proc = 1
        ndim = len(self.free_params)
        if sampler is None:
            if sampler_type == 'ensemble':
                sampler = emcee.EnsembleSampler(
                    nwalkers,
                    ndim,
                    _ComputeLnProbEval(self),
                    threads=num_proc,
                    a=sampler_a
                )
            elif sampler_type == 'pt':
                raise NotImplementedError("PTSampler not done yet!")
                sampler = emcee.PTSampler(
                    ntemps,
                    nwalkers,
                    ndim,
                    logl,
                    logp
                )
            else:
                raise NotImplementedError(
                    "Sampler type %s not supported!" % (sampler_type,)
                )
        if sampler.chain.size == 0:
            theta0 = self.hyperprior.random_draw(size=nwalkers).T
            theta0 = theta0[:, ~self.fixed_params]
        else:
            # Start from the stopping point of the previous chain:
            theta0 = sampler.chain[:, -1, :]

        sampler.run_mcmc(theta0, nsamp)
        if plot_posterior or plot_chains:
            flat_trace = sampler.chain[:, burn::thin, :]
            flat_trace = flat_trace.reshape((-1, flat_trace.shape[2]))
        
        if plot_posterior and plot_chains:
            plot_sampler(
                sampler,
                labels=['$%s$' % (l,) for l in self.free_param_names],
                burn=burn
            )
        else:
            if plot_posterior:
                triangle.corner(
                    flat_trace,
                    plot_datapoints=False,
                    labels=['$%s$' % (l,) for l in self.free_param_names]
                )
            if plot_chains:
                f = plt.figure()
                for k in xrange(0, ndim):
                    # a = f.add_subplot(3, ndim, k + 1)
                    # a.acorr(
                    #     sampler.flatchain[:, k],
                    #     maxlags=100,
                    #     detrend=plt.mlab.detrend_mean
                    # )
                    # a.set_xlabel('lag')
                    # a.set_title('$%s$ autocorrelation' % (self.free_param_names[k],))
                    a = f.add_subplot(ndim, 1, 0 * ndim + k + 1)
                    for chain in sampler.chain[:, :, k]:
                        a.plot(chain)
                    a.set_xlabel('sample')
                    a.set_ylabel('$%s$' % (self.free_param_names[k],))
                    a.set_title('$%s$ all chains' % (self.free_param_names[k],))
                    a.axvline(burn, color='r', linewidth=3, ls='--')
                    # a = f.add_subplot(2, ndim, 1 * ndim + k + 1)
                    # a.plot(flat_trace[:, k])
                    # a.set_xlabel('sample')
                    # a.set_ylabel('$%s$' % (self.free_param_names[k],))
                    # a.set_title('$%s$ flattened, burned and thinned chain' % (self.free_param_names[k],))
            
        return sampler
    
    def compute_from_MCMC(self, X, n=0, return_mean=True, return_std=True,
                          return_cov=False, return_samples=False, num_samples=1,
                          noise=False, samp_kwargs={}, sampler=None,
                          flat_trace=None, burn=0, thin=1, **kwargs):
        """Compute desired quantities from MCMC samples of the hyperparameter posterior.
        
        The return will be a list with a number of rows equal to the number of
        hyperparameter samples. The columns depend on the state of the boolean
        flags, but will be some subset of (mean, stddev, cov, samples), in that
        order. Samples will be the raw output of :py:meth:`draw_sample`, so you
        will need to remember to convert to an array and flatten if you want to
        work with a single sample.
        
        Parameters
        ----------
        X : array-like (`M`,) or (`M`, `num_dim`)
            The values to evaluate the Gaussian process at.
        n : non-negative int or list, optional
            The order of derivative to compute. For num_dim=1, this must be an
            int. For num_dim=2, this must be a list of ints of length 2.
            Default is 0 (don't take derivative).
        return_mean : bool, optional
            If True, the mean will be computed at each hyperparameter sample.
            Default is True (compute mean).
        return_std : bool, optional
            If True, the standard deviation will be computed at each
            hyperparameter sample. Default is True (compute stddev).
        return_cov : bool, optional
            If True, the covariance matrix will be computed at each
            hyperparameter sample. Default is True (compute stddev).
        return_samples : bool, optional
            If True, random sample(s) will be computed at each hyperparameter
            sample. Default is False (do not compute samples).
        num_samples : int, optional
            Compute this many samples if `return_sample` is True. Default is 1.
        noise : bool, optional
            If True, noise is included in the predictions and samples. Default
            is False (do not include noise).
        samp_kwargs : dict, optional
            If `return_sample` is True, the contents of this dictionary will be
            passed as kwargs to :py:meth:`draw_sample`.
        sampler : :py:class:`Sampler` instance or None, optional
            :py:class:`Sampler` instance that has already been run to the extent
            desired on the hyperparameter posterior. If None, a new sampler will
            be created with :py:meth:`sample_hyperparameter_posterior`. In this
            case, all extra kwargs will be passed on, allowing you to set the
            number of samples, etc. Default is None (create sampler).
        flat_trace : array-like (`nsamp`, `ndim`) or None, optional
            Flattened trace with samples of the free hyperparameters. If present,
            overrides `sampler`. This allows you to use a sampler other than the
            ones from :py:mod:`emcee`, or to specify arbitrary values you wish
            to evaluate the curve at. Note that this WILL be thinned and burned
            according to the following two kwargs. "Flat" refers to the fact
            that you must have combined all chains into a single one. Default is
            None (use `sampler`).
        burn : int, optional
            The number of samples to discard at the beginning of the chain.
            Default is 0.
        thin : int, optional
            Every `thin`-th sample is kept. Default is 1.
        num_proc : int, optional
            The number of processors to use for evaluation. This is used both
            when calling the sampler and when evaluating the Gaussian process.
            If None, the number of available processors will be used. If zero,
            evaluation will proceed in parallel. Default is to use all available
            processors.
        **kwargs : extra optional kwargs
            All additional kwargs are passed to
            :py:meth:`sample_hyperparameter_posterior`.
        
        Returns
        -------
        out : dict
            A dictionary having some or all of the fields 'mean', 'std', 'cov'
            and 'samp'. Each entry is a list of array-like. The length of this
            list is equal to the number of hyperparameter samples used, and the
            entries have the following shapes:
            
                ==== ====================
                mean (`M`,)
                std  (`M`,)
                cov  (`M`, `M`)
                samp (`M`, `num_samples`)
                ==== ====================
        """
        output_transform = kwargs.pop('output_transform', None)
        if flat_trace is None:
            if sampler is None:
                sampler = self.sample_hyperparameter_posterior(burn=burn, **kwargs)
                # If we create the sampler, we need to make sure we clean up its pool:
                sampler.pool.close()
                
            flat_trace = sampler.chain[:, burn::thin, :]
            flat_trace = flat_trace.reshape((-1, flat_trace.shape[2]))
        else:
            flat_trace = flat_trace[burn::thin, :]
        
        num_proc = kwargs.get('num_proc', multiprocessing.cpu_count())
        
        if num_proc > 1:
            pool = multiprocessing.Pool(processes=num_proc)
            try:
                res = pool.map(
                    _ComputeGPWrapper(
                        self,
                        X,
                        n,
                        return_mean,
                        return_std,
                        return_cov,
                        return_samples,
                        num_samples,
                        noise,
                        samp_kwargs,
                        output_transform
                    ),
                    flat_trace
                )
            finally:
                pool.close()
        else:
            res = map(
                _ComputeGPWrapper(
                    self,
                    X,
                    n,
                    return_mean,
                    return_std,
                    return_cov,
                    return_samples,
                    num_samples,
                    noise,
                    samp_kwargs,
                    output_transform
                    ),
                    flat_trace
                )
        out = dict()
        if return_mean:
            out['mean'] = [r['mean'] for r in res if r is not None]
        if return_std:
            out['std'] = [r['std'] for r in res if r is not None]
        if return_cov:
            out['cov'] = [r['cov'] for r in res if r is not None]
        if return_samples:
            out['samp'] = [r['samp'] for r in res if r is not None]
        return out
    
    def predict_MCMC(self, X, ddof=1, full_MC=False, rejection_func=None, **kwargs):
        """Make a prediction using MCMC samples.
        
        This is essentially a convenient wrapper of :py:meth:`compute_from_MCMC`,
        designed to act more or less interchangeably with :py:meth:`predict`.
        
        Computes the mean of the GP posterior marginalized over the
        hyperparameters using iterated expectations. If `return_std` is True,
        uses the law of total variance to compute the variance of the GP
        posterior marginalized over the hyperparameters. If `return_cov` is True,
        uses the law of total covariance to compute the entire covariance of the
        GP posterior marginalized over the hyperparameters. If both `return_cov`
        and `return_std` are True, then both the covariance matrix and standard
        deviation array will be returned.
        
        Parameters
        ----------
        X : array-like (`M`,) or (`M`, `num_dim`)
            The values to evaluate the Gaussian process at.
        ddof : int, optional
            The degree of freedom correction to use when computing the variance.
            Default is 1 (standard Bessel correction for unbiased estimate).
        return_std : bool, optional
            If True, the standard deviation is also computed. Default is True.
        full_MC : bool, optional
            Set to True to compute the mean and covariance matrix using Monte
            Carlo sampling of the posterior. The samples will also be returned
            if full_output is True. Default is False (don't use full sampling).
        rejection_func : callable, optional
            Any samples where this function evaluates False will be rejected,
            where it evaluates True they will be kept. Default is None (no
            rejection). Only has an effect if `full_MC` is True.
        ddof : int, optional
        **kwargs : optional kwargs
            All additional kwargs are passed directly to
            :py:meth:`compute_from_MCMC`.
        """
        return_std = kwargs.get('return_std', True)
        return_cov = kwargs.get('return_cov', False)
        if full_MC:
            kwargs['return_mean'] = False
            kwargs['return_std'] = False
            kwargs['return_cov'] = False
            kwargs['return_samples'] = True
        else:
            kwargs['return_mean'] = True
        return_samples = kwargs.get('return_samples', True)
        res = self.compute_from_MCMC(X, **kwargs)
        
        out = {}
        
        if return_samples:
            samps = scipy.asarray(scipy.hstack(res['samp']))
        
        if full_MC:
            if rejection_func:
                good_samps = []
                for samp in samps.T:
                    if rejection_func(samp):
                        good_samps.append(samp)
                if len(good_samps) == 0:
                    raise ValueError("Did not get any good samples!")
                samps = scipy.asarray(good_samps, dtype=float).T
            mean = scipy.mean(samps, axis=1)
            cov = scipy.cov(samps, rowvar=1, ddof=ddof)
            std = scipy.sqrt(scipy.diagonal(cov))
        else:
            means = scipy.asarray(res['mean'])
            mean = scipy.mean(means, axis=0)
            
            if 'cov' in res:
                covs = scipy.asarray(res['cov'])
                cov = scipy.mean(covs, axis=0) + scipy.cov(means, rowvar=0, ddof=ddof)
                std = scipy.sqrt(scipy.diagonal(cov))
            elif 'std' in res:
                vars_ = scipy.asarray(scipy.asarray(res['std']))**2
                std = scipy.sqrt(scipy.mean(vars_, axis=0) +
                                 scipy.var(means, axis=0, ddof=ddof))
        
        out['mean'] = mean
        if return_samples:
            out['samp'] = samps
        if return_std or return_cov:
            out['std'] = std
        if return_cov:
            out['cov'] = cov
        
        return out

class _ComputeGPWrapper(object):
    """Wrapper to allow parallel evaluation of means, covariances and random draws.
    
    Parameters
    ----------
    gp : :py:class:`GaussianProcess` instance
        The :py:class:`GaussianProcess` to wrap.
    X : array-like
        The evaluation locations to use. No pre-processing is performed: `X`
        will be passed directly to :py:meth:`predict` and/or :py:meth:`draw_sample`.
    n : int or array-like
        The derivative orders to use. No pre-processing is performed: `n` will
        be passed directly to :py:meth:`predict` and/or :py:meth:`draw_sample`.
    return_mean : bool
        If True, the mean will be computed.
    return_std : bool
        If True, the standard deviation will be computed.
    return_cov : bool
        If True, the covariance matrix will be computed.
    return_sample : bool
        If True, random sample(s) will be computed.
    num_samples : int
        If `return_sample` is True, this many random samples will be computed.
    noise : bool
        If True, noise will be included in the prediction and samples.
    samp_kwargs : dict
        The contents of this dictionary will be passed to :py:meth:`draw_sample`.
    """
    def __init__(self, gp, X, n, return_mean, return_std, return_cov,
                 return_sample, num_samples, noise, samp_kwargs, output_transform):
        self.gp = gp
        self.X = X
        self.n = n
        self.return_mean = return_mean
        self.return_std = return_std
        self.return_cov = return_cov
        self.return_sample = return_sample
        self.num_samples = num_samples
        self.noise = noise
        self.samp_kwargs = samp_kwargs
        self.full_output = return_cov or return_std or return_sample
        self.output_transform = output_transform
    
    def __call__(self, p_case):
        """Evaluate the desired quantities with free hyperparameters `p_case`.
        
        Returns a dict with some or all of the fields 'mean', 'cov', 'std', 'samp'
        """
        try:
            self.gp.update_hyperparameters(list(p_case))
            out = self.gp.predict(
                self.X,
                n=self.n,
                noise=self.noise,
                full_output=self.full_output,
                return_samples=self.return_sample,
                num_samples=self.num_samples,
                output_transform=self.output_transform
            )
            if not self.full_output:
                # If full output is True, return_mean must be the only True thing,
                # since otherwise this isn't computing anything!
                return {'mean': out}
            else:
                if not self.return_mean:
                    out.pop('mean')
                if not self.return_std:
                    out.pop('std')
                if not self.return_cov:
                    out.pop('cov')
        except Exception as e:
            out = None
            warnings.warn(
                "Encountered exception during evaluation of MCMC samples. "
                "Exception is:\n%s\nParams are:\n%s"
                % (
                    e,
                    str(list(p_case))
                )
            )
        return out

class _ComputeLnProbEval(object):
    """Helper class to allow emcee to sample in parallel.
    
    Parameters
    ----------
    gp : :py:class:`GaussianProcess` instance
        The :py:class:`GaussianProcess` instance to wrap.
    """
    def __init__(self, gp):
        self.gp = gp
    
    def __call__(self, x):
        """Return the log-probability of the given hyperparameters.
        
        Parameters
        ----------
        x : array-like
            The new hyperparameters.
        """
        return -1 * self.gp.update_hyperparameters(x.flatten())

class _OptimizeHyperparametersEval(object):
    """Helper class to support parallel random starts of MAP estimation of hyperparameters.
    
    Parameters
    ----------
    gp : :py:class:`GaussianProcess` instance
        Instance to wrap to allow parallel optimization of.
    opt_kwargs : dict
        Dictionary of keyword arguments to be passed to
        :py:func:`scipy.optimize.minimize`.
    """
    def __init__(self, gp, opt_kwargs):
        self.gp = gp
        self.opt_kwargs = opt_kwargs
    
    def __call__(self, samp):
        try:
            return scipy.optimize.minimize(
                self.gp.update_hyperparameters,
                samp,
                **self.opt_kwargs
            )
        except AttributeError:
            warnings.warn("scipy.optimize.minimize not available, defaulting "
                          "to fmin_slsqp.",
                          RuntimeWarning)
            return wrap_fmin_slsqp(
                self.gp.update_hyperparameters,
                samp,
                opt_kwargs=self.opt_kwargs
            )
        except:
            warnings.warn(
                "Minimizer failed, skipping sample. Error is: %s: %s. "
                "State of params is: %s"
                % (
                    sys.exc_info()[0],
                    sys.exc_info()[1],
                    str(self.gp.free_params),
                ),
                RuntimeWarning
            )
            return None

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
                    raise ValueError("Argument loc must be 'min', 'max' or an "
                                     "array of length %d" % self.gp.num_dim)
            else:
                loc = scipy.asarray(loc, dtype=float)
                if loc.shape == (self.gp.num_dim,):
                    self.loc = loc
                else:
                    raise ValueError("Argument loc must be 'min', 'max' or have "
                                     "length %d" % self.gp.num_dim)
        
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
                        raise ValueError("Each element in argument bounds must "
                                         "have length %d" % self.gp.num_dim)
                else:
                    bounds[k] = scipy.asarray(bounds[k], dtype=float)
                    if bounds[k].shape != (self.gp.num_dim,):
                        raise ValueError("Each element in argument bounds must "
                                         "have length %d" % self.gp.num_dim)
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
