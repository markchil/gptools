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

from .gaussian_process import *
from .error_handling import GPArgumentError

import multiprocessing
import scipy
import scipy.special
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.widgets as mplw
import matplotlib.gridspec as mplgs
import itertools
import collections

def parallel_compute_ll_matrix(gp, bounds, num_pts, num_proc=None):
    """Compute matrix of the log likelihood over the parameter space in parallel.
    
    Parameters
    ----------
    bounds : 2-tuple or list of 2-tuples with length equal to the number of free parameters
        Bounds on the range to use for each of the parameters. If a single
        2-tuple is given, it will be used for each of the parameters.
    num_pts : int or list of ints with length equal to the number of free parameters
        The number of points to use for each parameters. If a single int is
        given, it will be used for each of the parameters.
    num_proc : Positive int or None, optional
        Number of processes to run the parallel computation with. If set to
        None, ALL available cores are used. Default is None (use all available
        cores).
    
    Returns
    -------
    ll_vals : :py:class:`Array`
        The log likelihood for each of the parameter possibilities.
    param_vals : list of :py:class:`Array`
        The parameter values used.
    """
    if num_proc is None:
        num_proc = multiprocessing.cpu_count()
    
    present_free_params = scipy.concatenate((gp.k.free_params, gp.noise_k.free_params))
    
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
    
    pv_cases = list()
    gp_cases = list()
    num_pts_cases = list()
    for k in xrange(0, len(param_vals[0])):
        specific_param_vals = list(param_vals)
        specific_param_vals[0] = param_vals[0][k]
        pv_cases.append(specific_param_vals)
        
        gp_cases.append(GaussianProcess(gp.k, noise_k=gp.noise_k))
        gp_cases[-1].add_data(gp.X, gp.y, err_y=gp.err_y, n=gp.n)
        
        num_pts_cases.append(num_pts)
    
    pool =  multiprocessing.Pool(processes=num_proc)
    vals = scipy.asarray(
        pool.map(
            _compute_ll_matrix_wrapper,
            zip(gp_cases, pv_cases, num_pts_cases)
        )
    )
    pool.close()
    
    return (vals, param_vals)
    
def _compute_ll_matrix_wrapper(gppv):
    """Helper wrapper function to enable use of multiprocessing.map.
    
    Parameters
    ----------
    gppv : 3-tuple
        (`gp`, `param_vals`, `num_pts`)
    
    Returns
    -------
    vals : :py:class:`Array`
        Log likelihood evaluated at the parameters specified in `param_vals`.
    """
    return gppv[0]._compute_ll_matrix(1, gppv[1], gppv[2])

def slice_plot(*args, **kwargs):
    """Constructs a plot that lets you look at slices through a multidimensional array.
    
    Parameters
    ----------
    vals : :py:class:`Array`, (`M`, `N`, `P`, ...)
        Multidimensional array to visualize.
    x_vals_1 : :py:class:`Array`, (`M`,)
        Values along the first dimension.
    x_vals_2 : :py:class:`Array`, (`N`,)
        Values along the second dimension.
    x_vals_3 : :py:class:`Array`, (`P`,)
        Values along the third dimension.
        
        **...and so on. At least four arguments must be provided.**
    
    names : list of strings, optional
        Names for each of the parameters at hand. If None, sequential numerical
        identifiers will be used. Length must be equal to the number of
        dimensions of `vals`. Default is None.
    n : Positive int, optional
        Number of contours to plot. Default is 100.
    
    Returns
    -------
        f : :py:class:`Figure`
            The Matplotlib figure instance created.
    
    Raises
    ------
        GPArgumentError
            If the number of arguments is less than 4.
    """
    names = kwargs.get('names', None)
    n = kwargs.get('n', 100)
    num_axes = len(args) - 1
    if num_axes < 3:
        raise GPArgumentError("Must pass at least four arguments to slice_plot!")
    if num_axes != args[0].ndim:
        raise GPArgumentError("Number of dimensions of the first argument "
                              "must match the number of additional arguments "
                              "provided!")
    if names is None:
        names = [str(k) for k in range(2, num_axes)]
    f = plt.figure()
    height_ratios = [8]
    height_ratios += (num_axes - 2) * [1]
    gs = mplgs.GridSpec(num_axes - 2 + 1, 2, height_ratios=height_ratios, width_ratios=[8, 1])
    
    a_main = f.add_subplot(gs[0, 0])
    a_cbar = f.add_subplot(gs[0, 1])
    a_sliders = []
    for idx in xrange(0, num_axes - 2):
        a_sliders.append(f.add_subplot(gs[idx+1, :]))
    
    title = f.suptitle("")
    
    def update(val):
        """Update the slice shown.
        """
        a_main.clear()
        a_cbar.clear()
        idxs = [int(slider.val) for slider in sliders]
        vals = [args[k + 3][idxs[k]] for k in range(0, num_axes - 2)]
        descriptions = tuple(itertools.chain.from_iterable(itertools.izip(names[2:], vals)))
        fmt = "Slice" + (num_axes - 2) * ", %s: %f"
        title.set_text(fmt % descriptions)
        
        a_main.set_xlabel(names[1])
        a_main.set_ylabel(names[0])
        cs = a_main.contour(args[2], args[1], args[0][scipy.s_[:, :] + tuple(idxs)].squeeze(), n, vmin=args[0].min(), vmax=args[1].max())
        cbar = f.colorbar(cs, cax=a_cbar)
        cbar.set_label("LL")
        
        f.canvas.draw()
    
    idxs_0 = (num_axes - 2) * [0]
    sliders = []
    for idx in xrange(0, num_axes - 2):
        sliders.append(mplw.Slider(a_sliders[idx],
                                   '%s index' % names[idx + 2],
                                   0,
                                   len(args[idx + 3]) - 1,
                                   valinit=idxs_0[idx],
                                   valfmt='%d'))
        sliders[-1].on_changed(update)
    
    update(idxs_0)
    
    f.canvas.mpl_connect('key_press_event', lambda evt: arrow_respond(sliders[0], evt))
    
    return f

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

def unique_rows(arr):
    """Returns a copy of arr with duplicate rows removed.
    
    From Stackoverflow "Find unique rows in numpy.array."
    
    Parameters
    ----------
    arr : :py:class:`Array`, (`m`, `n`). The array to find the unique rows of.
    
    Returns
    -------
    unique : :py:class:`Array`, (`p`, `n`) where `p` <= `m`
        The array `arr` with duplicate rows removed.
    """
    b = scipy.ascontiguousarray(arr).view(
        scipy.dtype((scipy.void, arr.dtype.itemsize * arr.shape[1]))
    )
    try:
        dum, idx = scipy.unique(b, return_index=True)
    except TypeError:
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
    return arr[idx]

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

def compute_stats(vals, check_nan=False, robust=False, axis=1, plot_QQ=False):
    """Compute the average statistics (mean, std dev) for the given values.
    
    Parameters
    ----------
    vals : array-like, (`M`, `N`)
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
        Whether or not a QQ plot should be drawn for each channel. Default is
        False (do not draw QQ plots).
    
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
        raise NotImplementedError("Not implemented yet!")
    return (mean, std)
