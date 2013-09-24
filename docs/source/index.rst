.. gptools documentation master file, created by
   sphinx-quickstart on Tue Sep  3 12:06:03 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

gptools: Gaussian process regression with support for arbitrary derivatives
===========================================================================

Overview
--------

:py:mod:`gptools` is a Python package that provides a convenient, powerful and extensible implementation of Gaussian process regression (GPR). Central to :py:mod:`gptool`'s implementation is support for derivatives and their variances. A number of kernels are provided to allow many types of data to be fit:

* :py:class:`~gptools.kernel.noise.DiagonalNoiseKernel` implements homoscedastic noise. The noise is tied to a specific derivative order. This allows you to, for instance, have noise on your observations but have noiseless derivative constraints, or to have different noise levels for observations and derivatives. Note that you can also specify potentially heteroscedastic noise explicitly when adding data to the process.
* :py:class:`~gptools.kernel.squared_exponential.SquaredExponentialKernel` implements the SE kernel which is infinitely differentiable.
* :py:class:`~gptools.kernel.matern.MaternKernel` implements the entire Matern class of covariance functions, which are characterized by a hyperparameter :math:`\nu`. A process having the Matern kernel is only mean-square differentiable for derivative order :math:`n<\nu`.
* :py:class:`~gptools.kernel.rational_quadratic.RationalQuadraticKernel` implements the rational quadratic kernel, which is a scale mixture over SE kernels.

In all cases, these kernels have been constructed in a way to allow inputs of arbitrary dimension. Each dimension has a length scale hyperparameter that can be separately optimized over or held fixed. Arbitrary derivatives with respect to each dimension can be taken, including computation of the covariance for those observations.

Other kernels can be implemented by extending the :py:class:`~gptools.kernel.core.Kernel` class. Furthermore, kernels may be added or multiplied together to yield a new, valid kernel.

Contents
--------

.. toctree::
   :maxdepth: 4
   
   gptools

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

