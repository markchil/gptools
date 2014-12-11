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

"""Provides convenient utilities for working with the classes and results from :py:mod:`gptools`.

This module specifically contains utilities that need to interact directly with
the GaussianProcess object, and hence can present circular import problems when
incorporated in the main utils submodule.
"""

from __future__ import division

from .gaussian_process import GaussianProcess
from .error_handling import GPArgumentError

import multiprocessing
try:
    import matplotlib.pyplot as plt
    import matplotlib.widgets as mplw
    import matplotlib.gridspec as mplgs
except ImportError:
    import warnings
    warnings.warn("Could not import matplotlib. slice_plot will not be available.",
                  ImportWarning)
import itertools
import scipy
import copy

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
    ll_vals : array
        The log likelihood for each of the parameter possibilities.
    param_vals : list of array
        The parameter values used.
    """
    if num_proc is None:
        num_proc = multiprocessing.cpu_count()
    
    present_free_params = gp.free_params
    
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
        
        gp_cases += [copy.deepcopy(gp)]
        
        num_pts_cases.append(num_pts)
    
    pool =  multiprocessing.Pool(processes=num_proc)    
    try:
        vals = scipy.asarray(
            pool.map(
                _compute_ll_matrix_wrapper,
                zip(gp_cases, pv_cases, num_pts_cases)
            )
        )
    finally:
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
    vals : array
        Log likelihood evaluated at the parameters specified in `param_vals`.
    """
    return gppv[0]._compute_ll_matrix(1, gppv[1], gppv[2])

def slice_plot(*args, **kwargs):
    """Constructs a plot that lets you look at slices through a multidimensional array.
    
    Parameters
    ----------
    vals : array, (`M`, `D`, `P`, ...)
        Multidimensional array to visualize.
    x_vals_1 : array, (`M`,)
        Values along the first dimension.
    x_vals_2 : array, (`D`,)
        Values along the second dimension.
    x_vals_3 : array, (`P`,)
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
        cs = a_main.contour(
            args[2],
            args[1],
            args[0][scipy.s_[:, :] + tuple(idxs)].squeeze(),
            n,
            vmin=args[0].min(),
            vmax=args[1].max()
        )
        cbar = f.colorbar(cs, cax=a_cbar)
        cbar.set_label("LL")
        
        f.canvas.draw()
    
    idxs_0 = (num_axes - 2) * [0]
    sliders = []
    for idx in xrange(0, num_axes - 2):
        sliders.append(
            mplw.Slider(
                a_sliders[idx],
                '%s index' % names[idx + 2],
                0,
                len(args[idx + 3]) - 1,
                valinit=idxs_0[idx],
                valfmt='%d'
            )
        )
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
