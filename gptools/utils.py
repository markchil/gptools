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

import scipy
import scipy.special
import scipy.stats

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
