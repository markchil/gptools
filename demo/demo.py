# Copyright 2015 Mark Chilenski
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

"""This file contains a demonstration of the basic functionality of :py:mod:`gptools`.

Extensive comments are provided throughout, but refer to the documentation at
gptools.readthedocs.org for full details.

The script takes a while to run. When it is complete, you should have a total of
4 figures -- one with data and a mess of different fits, and three showing MCMC
output.
"""

import gptools
import cPickle as pkl
import scipy
import matplotlib.pyplot as plt
plt.ion()


### Load the test data:
# core_data contains density data as a function of r/a from the core of an
# Alcator C-Mod H-mode.
# These data are expected to be fit properly with a stationary covariance
# kernel.
with open('sample_data_core.pkl', 'rb') as f:
    core_data = pkl.load(f)
# edge_data contains density data as a function of r/a from the edge of the
# same Alcator C-Mod H-mode as core_data.
# When these datasets are combined, a non-stationary covariance kernel becomes
# necessary.
with open('sample_data_edge.pkl', 'rb') as f:
    edge_data = pkl.load(f)

### Make some figures to hold our results:
f = plt.figure()
a_val = f.add_subplot(2, 1, 1)
a_val.errorbar(core_data['X'], core_data['y'], yerr=core_data['err_y'], label='core data', fmt='.')
a_val.errorbar(edge_data['X'], edge_data['y'], yerr=edge_data['err_y'], label='edge data', fmt='.')
a_grad = f.add_subplot(2, 1, 2)

### Once your data are loaded from whatever their source is, you will perform
### the following steps:
# 1. Create the covariance kernel that describes the spatial structure of the
#    data, including a (hyper)prior that describes any prior knowledge you
#    have regarding its hyperparameters.
# 2. Create the Gaussian process itself, telling it what covariance kernel to
#    use.
# 3. Add your data to the Gaussian process.
# 4. Make an inference for the hyperparameters -- this can be done either as a
#    maximum a posteriori (MAP) estimate, or with Markov chain Monte Carlo
#    (MCMC) sampling.
# 5. Make a prediction.

### First, we must create a covariance kernel.
# A wide variety of covariance kernels is supported, including various
# algebraic mainpulations of the more fundamental kernels. We will use the most
# basic -- the 1d stationary squared exponential (SE) covariance kernel -- to
# fit the core data with. At the end, a more complicated case with a
# nonstationary kernel is used to fit the entire dataset.

# In order for the inference to work properly, an appropriate (hyper)prior must
# be set for the two hyperparameters of the SE kernel: the prefactor and the
# covariance length scale. (Note to plasma physicists: the covariance length
# scale is not in any way the same as the gradient scale length!) There are two
# levels of sophistication which can be employed here: specifying a list of
# tuples for the `param_bounds` keyword places a uniform prior over each
# hyperparameter whereas passing a :py:class:`JointPrior` instance for the
# `hyperprior` keyword allows you to specify a far more complicated prior
# distribution if needed.

# First, an example using `param_bounds`:
# The prefactor corresponds roughly to the typical range of variation of the
# data. It is usually sufficient to let it vary from 0 to about 5 or 10 times
# the maximum value in your data. The covariance length scale dictates how
# rapidly the data can change in space. For the present data, a good guess is
# that it can vary from 0 to 5. If you were fitting multivariate data, you would
# use the `num_dim` keyword to indicate how many dimensions there are. (The
# default is 1.)
k_SE = gptools.SquaredExponentialKernel(param_bounds=[(0, 20), (0, 5)])

# A much more powerful approach is to specify a joint (hyper)prior distribution
# for the hyperparameters. When :py:class:`JointPrior` instances are multiplied,
# the two priors are taken as being independent prior distributions. So, we can
# make a uniform prior on [0, 20] for the prefactor and a gamma prior with mode
# 1 and standard-deivation 0.7 for the covariance length scale as follows:
hp = gptools.UniformJointPrior(0, 20) * gptools.GammaJointPriorAlt(1, 0.7)
k_SE = gptools.SquaredExponentialKernel(hyperprior=hp)

### Now, we can move on to creating the Gaussian process itself:
# When creating a Gaussian process, you must specify the covariance kernel to
# use. Optionally, you may also specify a covariance kernel that you consider to
# be a "noise" component. This allows you to represent correlated noise and
# acts in addition to the heteroscedastic, uncorrelated noise you can specify
# when adding data to the Gaussian process. This formulation lets you choose
# whether or not noise is included when making a prediction. (But note that the
# heteroscedastic, uncorrelated noise specified when adding the data does not
# enter the prediction because there is not enough information to know how it
# behaves as a function of the input variables!) If you wanted to fit for
# homoscedastic, uncorrelated noise, first create a
# :py:class:`DiagonalNoiseKernel`:
k_noise = gptools.DiagonalNoiseKernel(noise_bound=[0, 5])
# Then create the Gaussian process:
gp_noise = gptools.GaussianProcess(k_SE, noise_k=k_noise)
# But, we have pretty good estimate of the noise that comes with our data, so
# for this example I will not be using a noise kernel (we will compare the two
# answers at the end). I create a new SE kernel instance to avoid issues with
# object references.
gp = gptools.GaussianProcess(
    gptools.SquaredExponentialKernel(
        hyperprior=gptools.UniformJointPrior(0, 20) *
                   gptools.GammaJointPriorAlt(1, 0.7)
    )
)

### Now we are ready to add some data to our Gaussian process:
# Local measurements are very simple to add using :py:meth:`add_data`. `err_y`
# is the standard deviation of the (assumed Gaussian) uncorrelated noise on
# each data point. This noise can be heteroscedastic. If `err_y` is not given
# the data are taken to be exact and the curve will pass exactly through them,
# unless you have also specified a noise kernel.
gp.add_data(core_data['X'], core_data['y'], err_y=core_data['err_y'])
# You may also add derivative data using the `n` keyword. For instance, we can
# force the slope at the X=0 to be zero as follows:
gp.add_data(0, 0, n=1)
# It is also possible to add linearly-transformed data such as line integrals
# and volume averages using the `T` keyword. See the documentation for more
# details.

# If we didn't have good noise estimates and were trying to estimate the noise
# from the data, the above commands would simply be:
gp_noise.add_data(core_data['X'], core_data['y'])
gp_noise.add_data(0, 0, n=1)

### Now we need to figure out values for the hyperparameters:
# There are two levels of sophistication supported. The most basic approach is
# to find the maximum a posteriori (MAP) estimate -- the single set of values
# for the hyperparameters which is most likely given the data. In many cases
# this is acceptable. But, this will often underpredict the undertainty in the
# fitted curve. So, if you care about uncertainties you may need to use Markov
# chain Monte Carlo to marginalize over the hyperparameters. This is quite a bit
# more time-consuming, but gives you a complete picture of your posterior
# distribution.

# The MAP estimate conducts a very simple-minded global maximum search by
# starting many local optimizers at locations distributed according to your
# (hyper)prior distribution. This can be run in parallel, so if you have access
# to a multicore machine, use it. The number of starts is dictated by the
# `random_starts` keyword. By default one start per available processor is used.
# If you have a small number of cores and/or are getting bad fits, try
# increasing this number. (But note that it could be telling you that your
# parameter space is poorly-behaved, and you should probably consider trying
# MCMC!)
gp.optimize_hyperparameters(verbose=True)
# If we do this for the one with noise, we can compare the result:
gp_noise.optimize_hyperparameters(verbose=True)

# Following the optimization, the optimized values for the hyperparameters are
# stored in gp.params. This is a somewhat complicated :py:class:`CombinedBounds`
# object which lets you propagate changes back to the individual objects. View
# the parameters as follows:
print(gp.params[:])
print(gp_noise.params[:])
# You should have gotten something like:
# [1.8849006111246833, 0.97760159723344708, 0.0]
# [1.7095365754195335, 1.222639837707701, 0.12181881916114756]
# There may be small variations because of the random nature of the multistart
# optimization approach used. Note the extra 0 on the first set of
# hyperparameters -- this is because the noise is fixed to zero there. It is
# very reassuring that both converged to similar values for the prefactor and
# the covariance length scale, and that the fitted (homoscedastic) noise is
# comparable to the mean noise in `core_data['err_y']`.

### We'll come back to do the MCMC estimates in a bit -- first let's make some
### predictions with the trained GP.
# First, define a grid to evaluate on:
X_star = scipy.linspace(0, 1.1, 400)
# Now, we can make a prediction of the mean and standard deviation of the fit:
y_star, err_y_star = gp.predict(X_star)
# And we can plot this with our data:
gptools.univariate_envelope_plot(
    X_star,
    y_star,
    err_y_star,
    label='core, heteroscedastic noise, MAP',
    ax=a_val
)
# And we can do the same for our case where we inferred the noise:
y_star_noise, err_y_star_noise = gp_noise.predict(X_star)
gptools.univariate_envelope_plot(
    X_star,
    y_star_noise,
    err_y_star_noise,
    label='core, inferred noise, MAP',
    ax=a_val
)

# We can make predictions of the first derivative simply by using the `n`
# keyword, just like when adding data:
grad_y_star, err_grad_y_star = gp.predict(X_star, n=1)
gptools.univariate_envelope_plot(
    X_star,
    grad_y_star,
    err_grad_y_star,
    label='core, heteroscedastic noise, MAP',
    ax=a_grad
)
grad_y_star_noise, err_grad_y_star_noise = gp_noise.predict(X_star, n=1)
gptools.univariate_envelope_plot(
    X_star,
    grad_y_star_noise,
    err_grad_y_star_noise,
    label='core, inferred noise, MAP',
    ax=a_grad
)

# You can also make predictions of linearly-transformed quantities (line
# integrals), volume averages, etc. using the `output_transform` keyword, see
# the manual for more details.

# If we wanted to get the covariances between the values and derivatives, we
# should predict them at the same time by passing an array for `n`:
out = gp.predict(
    scipy.concatenate((X_star, X_star)),
    n=scipy.concatenate((scipy.zeros_like(X_star), scipy.ones_like(X_star))),
    full_output=True
)
# `out` contains the mean under key `mean`, the standard deviation under key
# `std` and the full covariance matrix under key `cov`.

# Another popular way of visualizing the posterior uncertainty is to make
# random draws from the posterior for the fitted curve. Once the
# hyperparameters have been selected with a MAP estimate (or otherwise set to a
# fixed value), this is accomplished as follows:
y_samp = gp.draw_sample(X_star, num_samp=10)
a_val.plot(X_star, y_samp, color='r', alpha=0.5)

# When a noise kernel has been used, this can be used to make synthetic data,
# including noise:
y_synth = gp_noise.draw_sample(scipy.sort(core_data['X']), num_samp=10, noise=True)
a_val.plot(scipy.sort(core_data['X']), y_synth, 'c.', alpha=0.5)
# This also supports the `n` keyword, just like `predict` does.

### The errorbars on the gradient are often underestimated by using a MAP
### estimate -- let's do it right, using MCMC.
# This parallelizes quite nicely, so try to run it on a multicore machine if
# that is available. The easiest way to get the posterior mean and standard
# deviation of the profile is with the `use_MCMC` keyword to the `predict`
# method. I get the value and the gradient at the same time so I only have to
# run the sampler to sample the hyperparameters once. `nsamp` tells it to take
# 200 samples from each chain (walker) it runs in parallel. `burn` tells it to
# throw away the first 100 samples from each chain before computing any
# profiles. `thin` tells it to only keep every tenth sample from each chain
# when computing the profiles. It takes some skill to spot when the Markov
# chains have settled down (and hence what to set `burn` to). See the
# documentation on `sample_hyperparameter_posterior` for more details.
y_grad_y_star, std_y_grad_y_star = gp.predict(
    scipy.concatenate((X_star, X_star)),
    n=scipy.concatenate((scipy.zeros_like(X_star), scipy.ones_like(X_star))),
    use_MCMC=True,
    nsamp=200,
    burn=100,
    thin=10,
    plot_posterior=True,
    plot_chains=True
)
gptools.univariate_envelope_plot(
    X_star,
    y_grad_y_star[:len(X_star)],
    std_y_grad_y_star[:len(X_star)],
    label='core, heteroscedastic noise, MCMC',
    ax=a_val
)
gptools.univariate_envelope_plot(
    X_star,
    y_grad_y_star[len(X_star):],
    std_y_grad_y_star[len(X_star):],
    label='core, heteroscedastic noise, MCMC',
    ax=a_grad
)

# It is usually preferable to get the MCMC sampler tuned up and burned in first,
# then to compute the profiles as a separate step. This is accomplished as
# follows:
sampler = gp_noise.sample_hyperparameter_posterior(
    nsamp=200,
    burn=100,
    plot_posterior=True,
    plot_chains=True
)
# With the sampler already computed, the computational advantage of predicting
# the values and gradients at the same time is no longer there:
y_star, std_y_star = gp.predict(
    X_star,
    use_MCMC=True,
    sampler=sampler,
    burn=100,
    thin=10
)
gptools.univariate_envelope_plot(
    X_star,
    y_star,
    std_y_star,
    label='core, inferred noise, MCMC',
    ax=a_val
)
grad_y_star, std_grad_y_star = gp.predict(
    X_star,
    n=1,
    use_MCMC=True,
    sampler=sampler,
    burn=100,
    thin=10
)
gptools.univariate_envelope_plot(
    X_star,
    grad_y_star,
    std_grad_y_star,
    label='core, inferred noise, MCMC',
    ax=a_grad
)

# We can again repeat our trick of drawing random samples from the posterior for
# the fitted curve. This is a two-step sampling process: first, samples of the
# hyperparameters are drawn using MCMC. Then, for each value of the
# hyperparameters which we keep (i.e., after burning/thinning), we draw one or
# more samples. Note that this cannot be done with `draw_sample`, as that method
# assumes a single, fixed value for the hyperparmeters has been set. Instead, we
# use `predict` in the following way:
out = gp.predict(
    X_star,
    use_MCMC=True,
    sampler=sampler,
    burn=100,
    thin=100,
    full_output=True,
    return_samples=True,
    num_samples=1
)
# This will draw 1 sample of the fit for every sample of hyperparameters which
# is kept.
a_val.plot(X_star, out['samp'], color='y', alpha=0.1)

### When the edge data are incorporated, a stationary kernel such as the SE is
### no longer appropriate:
# Either the region of rapid change will drive the fit to short values, or the
# gradual region will cause the rapid change to be oversmoothed.
# :py:class:`GibbsKernel1dTanh` was designed to fit nonstationary data like
# this where there is a smooth region and a rough region. An arbitrary length
# scale function can be selected by following this template. In addition, more
# powerful input warpings are provided in the :py:mod:`warping` submodule. See
# the manual for more details.
hp = (
    gptools.UniformJointPrior([[0.0, 20.0]]) *
    gptools.GammaJointPriorAlt([1.0, 0.5, 0.0, 1.0], [0.3, 0.25, 0.1, 0.1])
)
k_gibbs = gptools.GibbsKernel1dTanh(hyperprior=hp)
gp = gptools.GaussianProcess(k_gibbs)
gp.add_data(core_data['X'], core_data['y'], err_y=core_data['err_y'])
gp.add_data(edge_data['X'], edge_data['y'], err_y=edge_data['err_y'])
gp.add_data(0, 0, n=1)
gp.optimize_hyperparameters(verbose=True)
y_star, std_y_star = gp.predict(X_star)
gptools.univariate_envelope_plot(
    X_star,
    y_star,
    std_y_star,
    label='whole profile, Gibbs+tanh kernel',
    ax=a_val
)
grad_y_star, std_grad_y_star = gp.predict(X_star, n=1)
gptools.univariate_envelope_plot(
    X_star,
    grad_y_star,
    std_grad_y_star,
    label='whole profile, Gibbs+tanh kernel',
    ax=a_grad
)
# This could (and usually should!) be done with MCMC sampling exactly as shown
# above.

### Another way of obtaining nonstationarity is thorugh a mean function:
# In effect, this allows you to perform a Bayesian nonlinear parameteric
# regression with a GP to fit the residuals. This is accomplished by passing a
# :py:class:`MeanFunction` instance when creating the Gaussian process. For this
# example, I will use the popular mtanh parameterization to fit the entire
# density profile, with a SE kernel for he residuals.
mu = gptools.MtanhMeanFunction1d()
hp = gptools.UniformJointPrior(0, 20) * gptools.GammaJointPriorAlt(1, 0.7)
k_SE = gptools.SquaredExponentialKernel(hyperprior=hp)
gp = gptools.GaussianProcess(k_SE, mu=mu)
gp.add_data(core_data['X'], core_data['y'], err_y=core_data['err_y'])
gp.add_data(edge_data['X'], edge_data['y'], err_y=edge_data['err_y'])
gp.add_data(0, 0, n=1)
gp.optimize_hyperparameters(verbose=True)
y_star, std_y_star = gp.predict(X_star)
gptools.univariate_envelope_plot(
    X_star,
    y_star,
    std_y_star,
    label='whole profile, mtanh mean function, SE kernel, MAP',
    ax=a_val
)
grad_y_star, std_grad_y_star = gp.predict(X_star, n=1)
gptools.univariate_envelope_plot(
    X_star,
    grad_y_star,
    std_grad_y_star,
    label='whole profile, mtanh mean function, SE kernel, MAP',
    ax=a_grad
)
# You can even take a gander at the entire posterior distribution for the
# parameters/hyperparameters:
sampler = gp.sample_hyperparameter_posterior(
    nsamp=500,
    burn=400,
    plot_posterior=True,
    plot_chains=True
)
# Examination of the posterior distribution tells us that `b` (the pedestal
# foot) is more or less unconstrained. Therefore, better prior information or
# more data are required to make quantitative statements about that parameter --
# and this extra uncertainty needs to be included in the fit.
# Evaluating the fit from the sampler proceeds exactly as described above:
y_star, std_y_star = gp.predict(
    X_star,
    use_MCMC=True,
    sampler=sampler,
    burn=100,
    thin=10
)
gptools.univariate_envelope_plot(
    X_star,
    y_star,
    std_y_star,
    label='whole profile, mtanh mean function, SE kernel, MCMC',
    ax=a_val
)
grad_y_star, std_grad_y_star = gp.predict(
    X_star,
    n=1,
    use_MCMC=True,
    sampler=sampler,
    burn=100,
    thin=10
)
gptools.univariate_envelope_plot(
    X_star,
    grad_y_star,
    std_grad_y_star,
    label='whole profile, mtanh mean function, SE kernel, MCMC',
    ax=a_grad
)

a_val.legend(loc='best')
f.canvas.draw()
