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

"""Subpackage containing a variety of covariance kernels and associated helpers.
"""

from __future__ import division

from .core import Kernel, BinaryKernel, SumKernel, ProductKernel, ChainRuleKernel
from .matern import MaternKernel
from .noise import DiagonalNoiseKernel, ZeroKernel
from .squared_exponential import SquaredExponentialKernel
from .rational_quadratic import RationalQuadraticKernel
from .gibbs import (GibbsKernel1dtanh, GibbsKernel1dSpline, GibbsKernel1dGauss,
                    GibbsFunction1d, tanh_warp, spline_warp, gauss_warp)