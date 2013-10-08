gptools
=======

Gaussian processes with arbitrary derivative constraints and predictions.

Univariate, stationary kernels have been characterized fairly well at this point. The univariate non-stationary ("Gibbs") kernel is fairly far along for tanh and polynomial "bucket" warping. Has been written with handling multivariate data in mind, but this has not been tested extensively yet.

Developed and tested using Python 2.7 and scipy 0.12.0. Catches have been included to enable use with scipy 0.10.1. May work with other versions, but it has not been tested under such configurations.

Full package documentation is located at http://gptools.readthedocs.org/

A printable PDF is available at https://media.readthedocs.org/pdf/gptools/latest/gptools.pdf