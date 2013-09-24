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

""":py:mod:`gptools` - Gaussian process regression with support for arbitrary derivatives
"""

from __future__ import division

from .gaussian_process import GaussianProcess, Constraint
from .error_handling import GPArgumentError
from .kernel import (Kernel, BinaryKernel, SumKernel, ProductKernel,
                     ChainRuleKernel, MaternKernel, DiagonalNoiseKernel,
                     ZeroKernel, SquaredExponentialKernel, RationalQuadraticKernel,
                     GibbsKernel1dtanh, GibbsKernel1dSpline, GibbsKernel1dGauss,
                     GibbsFunction1d, tanh_warp, spline_warp, gauss_warp)
from .utils import (parallel_compute_ll_matrix, slice_plot, arrow_respond,
                    incomplete_bell_poly, generate_set_partition_strings,
                    generate_set_partitions, unique_rows)