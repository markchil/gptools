from __future__ import division
import gptools
import eqtools
import MDSplus
import time
import scipy
import scipy.stats
import numpy.random
import matplotlib
matplotlib.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt
plt.ion()
#plt.close('all')

shot = 1101014006
# Start and end times of flat top:
flat_start = 0.5
flat_stop = 1.5

efit_tree = eqtools.CModEFITTree(shot)
t_EFIT = efit_tree.getTimeBase()

electrons = MDSplus.Tree('electrons', shot)

# Get core TS data:
N_Te_TS = electrons.getNode(r'\electrons::top.yag_new.results.profiles:te_rz')

t_Te_TS = N_Te_TS.dim_of().data()

# Only keep points that are in the RF flat top:
ok_idxs = (t_Te_TS >= flat_start) & (t_Te_TS <= flat_stop)
t_Te_TS = t_Te_TS[ok_idxs]

Te_TS = N_Te_TS.data()[:, ok_idxs]
dev_Te_TS = electrons.getNode(r'yag_new.results.profiles:te_err').data()[:, ok_idxs]

Z_CTS = electrons.getNode(r'yag_new.results.profiles:z_sorted').data()
R_CTS = electrons.getNode(r'yag.results.param:r').data() * scipy.ones_like(Z_CTS)
t_grid, Z_grid = scipy.meshgrid(t_Te_TS, Z_CTS)
t_grid, R_grid = scipy.meshgrid(t_Te_TS, R_CTS)
R_mid_CTS = efit_tree.rz2rmid(R_grid, Z_grid, t_grid, each_t=False)

# Get edge data:
N_Te_ETS = electrons.getNode(r'yag_edgets.results:te')

t_Te_ETS = N_Te_ETS.dim_of().data()
# Assume ETS is on same timebase as CTS and use indices from above:
t_Te_ETS = t_Te_ETS[ok_idxs]

Te_ETS = N_Te_ETS.data()[:, ok_idxs] / 1e3
dev_Te_ETS = electrons.getNode(r'yag_edgets.results:te:error').data()[:, ok_idxs] / 1e3

Z_ETS = electrons.getNode(r'yag_edgets.data:fiber_z').data()
R_ETS = electrons.getNode(r'yag.results.param:R').data() * scipy.ones_like(Z_ETS)
t_grid, Z_grid = scipy.meshgrid(t_Te_ETS, Z_ETS)
t_grid, R_grid = scipy.meshgrid(t_Te_ETS, R_ETS)
R_mid_ETS = efit_tree.rz2rmid(R_grid, Z_grid, t_grid, each_t=False)

# Flag bad points for exclusion:
Te_ETS[(Te_ETS == 0) & (dev_Te_ETS == 1)] = scipy.nan

# Get FRCECE data:
Te_FRC = []
R_mid_FRC = []
for k in xrange(0, 32):
    N = electrons.getNode(r'frcece.data.eces%02d' % (k + 1,))
    Te_FRC.append(N.data())
    N_R = electrons.getNode(r'frcece.data.rmid_%02d' % (k + 1,))
    R_mid_FRC.append(N_R.data().flatten())
# Assume all slow channels are on the same timebase:
t_FRC = N.dim_of().data()
# The radius is given on the EFIT timebase, so must be handled separately:
t_R_FRC = N_R.dim_of().data()
# Remove points outside of range of interest:
ok_idxs = (t_FRC >= flat_start) & (t_FRC <= flat_stop)
t_FRC = t_FRC[ok_idxs]
Te_FRC = scipy.asarray(Te_FRC, dtype=float)[:, ok_idxs]
# Handling the radius like this will be fine for shot-average, but needs to be
# interpolated to handle single-frame:
ok_idxs = (t_R_FRC >= flat_start) & (t_R_FRC <= flat_stop)
t_R_FRC = t_R_FRC[ok_idxs]
R_mid_FRC = scipy.asarray(R_mid_FRC, dtype=float)[:, ok_idxs]

# Get GPC2 data:
N_GPC2 = electrons.getNode('gpc_2.results.gpc2_te')
Te_GPC2 = N_GPC2.data()
t_GPC2 = N_GPC2.dim_of().data()
ok_idxs = (t_GPC2 >= flat_start) & (t_GPC2 <= flat_stop)
t_GPC2 = t_GPC2[ok_idxs]
Te_GPC2 = Te_GPC2[:, ok_idxs]
N_R = electrons.getNode('gpc_2.results.radii')
R_mid_GPC2 = N_R.data()
t_R_GPC2 = N_R.dim_of().data()
ok_idxs = (t_R_GPC2 >= flat_start) & (t_R_GPC2 <= flat_stop)
t_R_GPC2 = t_R_GPC2[ok_idxs]
R_mid_GPC2 = R_mid_GPC2[:, ok_idxs]

# Get magnetic axis location:
R_mag = efit_tree.getMagR()
R_mag_TS = scipy.interpolate.InterpolatedUnivariateSpline(t_EFIT, R_mag)(t_Te_TS)
R_mag_mean = scipy.mean(R_mag)
R_mag_std = scipy.std(R_mag)

# Get LCFS outboard midplane location:
R_out = efit_tree.getRmidOut()
R_out_TS = scipy.interpolate.InterpolatedUnivariateSpline(t_EFIT, R_out)(t_Te_TS)
R_out_mean = scipy.mean(R_out)
R_out_std = scipy.std(R_out)

# Compute weighted mean and weighted corected sample standard deviation:
# # Single time slice:
# idx = 44
# Te_TS_w = Te_TS[:, idx]
# dev_Te_TS_w = dev_Te_TS[:, idx]
# R_mid_w = R_mid_CTS[:, idx]
# 
# Te_ETS_w = Te_ETS[:, idx]
# good_idxs = ~scipy.isnan(Te_ETS_w)
# Te_ETS_w = Te_ETS_w[good_idxs]
# dev_Te_ETS_w = dev_Te_ETS[:, idx]
# dev_Te_ETS_w = dev_Te_ETS_w[good_idxs]
# R_mid_ETS_w = R_mid_ETS[:, idx]
# R_mid_ETS_w = R_mid_ETS_w[good_idxs]
# 
# R_mag_mean = R_mag[idx]
# R_mag_std = 0.0
# R_out_mean = R_out[idx]
# R_out_std = 0.0

# Average over entire data set:
Te_TS_w = scipy.mean(Te_TS, axis=1)
dev_Te_TS_w = scipy.std(Te_TS, axis=1)
R_mid_w = scipy.mean(R_mid_CTS, axis=1)
dev_R_mid_w = scipy.std(R_mid_CTS, axis=1)

Te_ETS_w = scipy.stats.nanmean(Te_ETS, axis=1)
dev_Te_ETS_w = scipy.stats.nanstd(Te_ETS, axis=1)
R_mid_ETS_w = scipy.mean(R_mid_ETS, axis=1)
dev_R_mid_ETS_w = scipy.std(R_mid_ETS, axis=1)

Te_FRC_w = scipy.mean(Te_FRC, axis=1)
dev_Te_FRC_w = scipy.std(Te_FRC, axis=1)
R_mid_FRC_w = scipy.mean(R_mid_FRC, axis=1)
dev_R_mid_FRC_w = scipy.std(R_mid_FRC, axis=1)
# Get rid of clearly too small points (Why do these happen?)
good_idxs = (Te_FRC_w >= 0.1)
Te_FRC_w = Te_FRC_w[good_idxs]
dev_Te_FRC_w = dev_Te_FRC_w[good_idxs]
R_mid_FRC_w = R_mid_FRC_w[good_idxs]
dev_R_mid_FRC_w = dev_R_mid_FRC_w[good_idxs]

Te_GPC2_w = scipy.mean(Te_GPC2, axis=1)
dev_Te_GPC2_w = scipy.std(Te_GPC2, axis=1)
R_mid_GPC2_w = scipy.mean(R_mid_GPC2, axis=1)
dev_R_mid_GPC2_w = scipy.std(R_mid_GPC2, axis=1)

# # Use entire data set, taking every skip-th point:
# skip = 1
# R_mid_w = R_mid_CTS.flatten()[::skip]
# Te_TS_w = Te_TS.flatten()[::skip]
# dev_Te_TS_w = dev_Te_TS.flatten()[::skip]
# dev_Te_TS_w = scipy.sqrt((dev_Te_TS_w)**2.0 +
#                          (scipy.repeat(scipy.std(Te_TS, axis=1), Te_TS.shape[1])[::skip])**2)
# 
# Te_ETS_w = Te_ETS.flatten()[::skip]
# good_idxs = ~scipy.isnan(Te_ETS_w)
# Te_ETS_w = Te_ETS_w[good_idxs]
# R_mid_ETS_w = R_mid_ETS.flatten()[::skip]
# R_mid_ETS_w = R_mid_ETS_w[good_idxs]
# dev_Te_ETS_w = dev_Te_ETS.flatten()[::skip]
# dev_Te_ETS_w = dev_Te_ETS_w[good_idxs]
# dev_Te_ETS_w = scipy.sqrt(((dev_Te_ETS).flatten()[::skip])**2.0 +
#                           (scipy.repeat(scipy.std(Te_ETS, axis=1), Te_ETS.shape[1])[::skip])**2)

# Set kernel:
# k = gptools.SquaredExponentialKernel(1,
#                                      initial_params=[1, 0.15],
#                                      fixed_params=[False, False],
#                                      param_bounds=[(0.0, 1000.0), (0.01, 1.0)])
# k = gptools.MaternKernel(1,
#                          initial_params=[1, 3.0/2.0, 0.15],
#                          fixed_params=[False, False, False],
#                          param_bounds=[(0.0, 1000.0), (0.51, 10.0), (0.01, 1.0)])
# k = gptools.RationalQuadraticKernel(1,
#                                     initial_params=[1, 20.0, 0.15],
#                                     fixed_params=[False, False, False],
#                                     param_bounds=[(0.0, 1000.0), (0.001, 100.0), (0.01, 1.0)],
#                                     enforce_bounds=True)
k = gptools.GibbsKernel1dtanh(
    initial_params=[1.88, 0.09655, 0.05637, 0.002941, 0.8937],
    fixed_params=[False, False, False, False, False],
    param_bounds=[(0.0, 1000.0), (0.01, 10.0), (0.0001, 1.0), (0.0001, 0.1), (0.88, 0.91)],
    num_proc=0,
    enforce_bounds=True
)

# Set noise kernel:
nk = gptools.DiagonalNoiseKernel(1, n=0, initial_noise=0.0, fixed_noise=True, noise_bound=(0.0, 10.0))

# Create and populate GP:
gp = gptools.GaussianProcess(k, noise_k=nk)
gp.add_data(R_mid_w, Te_TS_w, err_y=dev_Te_TS_w)
gp.add_data(R_mid_ETS_w, Te_ETS_w, err_y=dev_Te_ETS_w)
gp.add_data(R_mid_FRC_w, Te_FRC_w, err_y=dev_Te_FRC_w)
gp.add_data(R_mid_GPC2_w, Te_GPC2_w, err_y=dev_Te_GPC2_w)
gp.add_data(R_mag_mean, 0, n=1)

# Make constraint functions:
def l_cf(params):
    return params[1] - params[2]

class pos_cf(object):
    """Constraint to force the parameter in slot idx to be positive.
    
    Parameters
    ----------
    idx : non-negative int
        The index of the parameter to constrain.
    """
    def __init__(self, idx):
        self.idx = idx
    
    def __call__(self, params):
        """Evaluate the constraint.
        
        Parameters
        ----------
        params : list or other 1d array-like
            The parameters of the model.
        
        Returns
        -------
        Element `idx` of `params`.
        """
        return params[self.idx]

# Optimize hyperparameters:
opt_start = time.time()
gp.optimize_hyperparameters(
    method='SLSQP',
    verbose=True,
    opt_kwargs={
        'bounds': (k + nk).free_param_bounds,
        'constraints': (
            # {'type': 'ineq', 'fun': pos_cf(0)},
            # {'type': 'ineq', 'fun': pos_cf(1)},
            # {'type': 'ineq', 'fun': pos_cf(2)},
            # {'type': 'ineq', 'fun': pos_cf(3)},
            # {'type': 'ineq', 'fun': pos_cf(4)},
            # {'type': 'ineq', 'fun': pos_cf(5)},
            # {'type': 'ineq', 'fun': l_cf},
            # {'type': 'ineq', 'fun': gptools.Constraint(gp, n=1, type_='lt', loc='max')},
            # {'type': 'ineq', 'fun': gptools.Constraint(gp)},
        )
    }
)
opt_elapsed = time.time() - opt_start

# Make predictions:
Rstar = scipy.linspace(R_mag_mean, R_mid_ETS_w.max(), 24*5)

mean_start = time.time()
mean, cov = gp.predict(Rstar, noise=False)
mean_elapsed = time.time() - mean_start
mean = scipy.asarray(mean).flatten()
std = scipy.sqrt(scipy.diagonal(cov))

meand_start = time.time()
meand, covd = gp.predict(Rstar, noise=False, n=1)
meand_elapsed = time.time() - meand_start
meand = scipy.asarray(meand).flatten()
stdd = scipy.sqrt(scipy.diagonal(covd))

print("Optimization took: %.2fs\nMean prediction took: %.2fs\nGradient prediction took: %.2fs\n"  % (opt_elapsed, mean_elapsed, meand_elapsed))

"""meandd, covdd = gp.predict(Rstar, noise=False, n=2)
meandd = scipy.asarray(meandd).flatten()
stddd = scipy.sqrt(scipy.diagonal(covdd))"""


meand_approx = scipy.gradient(mean, Rstar[1] - Rstar[0])
meandd_approx = scipy.gradient(meand, Rstar[1] - Rstar[0])

f = plt.figure()

# f.suptitle('Univariate GPR on TS data')
f.suptitle('Univariate GPR on time-averaged data')
# f.suptitle('Univariate GPR on single frame of TS data, $t=%.2fs$' % t_Te_TS[idx])
#f.suptitle('With slope constraint')

a1 = f.add_subplot(3, 1, 1)
a1.plot(Rstar, mean, 'k', linewidth=3, label='mean')
a1.fill_between(Rstar, mean-std, mean+std, alpha=0.375, facecolor='k')
a1.errorbar(R_mid_w, Te_TS_w, xerr=dev_R_mid_w, yerr=dev_Te_TS_w, fmt='r.', label='CTS')
a1.errorbar(R_mid_ETS_w, Te_ETS_w, xerr=dev_R_mid_ETS_w, yerr=dev_Te_ETS_w, fmt='m.', label='ETS')
a1.errorbar(R_mid_FRC_w, Te_FRC_w, xerr=dev_R_mid_FRC_w, yerr=dev_Te_FRC_w, fmt='b.', label='FRC')
a1.errorbar(R_mid_GPC2_w, Te_GPC2_w, xerr=dev_R_mid_GPC2_w, yerr=dev_Te_GPC2_w, fmt='g.', label='GPC2')
a1.axvline(x=R_mag_mean, color='r', label='$R_{mag}$')
a1.axvspan(R_mag_mean-R_mag_std, R_mag_mean+R_mag_std, alpha=0.375, facecolor='r')
a1.axvline(x=R_out_mean, color='g', label='$R_{out}$')
a1.axvspan(R_out_mean-R_out_std, R_out_mean+R_out_std, alpha=0.375, facecolor='g')
a1.legend(loc='best')
#a1.set_xlabel('$R$ [m]')
a1.get_xaxis().set_visible(False)
a1.set_ylim(0, 4.5)
a1.set_ylabel('$T_{e}$ [keV]')

a2 = f.add_subplot(3, 1, 2, sharex=a1)
a2.plot(Rstar, meand, 'k', linewidth=3)
#a2.plot(Rstar, meand_approx)
a2.fill_between(Rstar, (meand-stdd), (meand+stdd), alpha=0.375, facecolor='k')
a2.axvline(x=R_mag_mean, color='r')
a2.axvspan(R_mag_mean-R_mag_std, R_mag_mean+R_mag_std, alpha=0.375, facecolor='r')
a1.axvline(x=R_out_mean, color='g')
a1.axvspan(R_out_mean-R_out_std, R_out_mean+R_out_std, alpha=0.375, facecolor='g')
# a2.set_xlabel('$R$ [m]')
a2.get_xaxis().set_visible(False)
a2.set_ylabel('$dT_{e}/dR$\n[keV/m]')

a3 = f.add_subplot(3, 1, 3, sharex=a1)
a3.plot(Rstar, gp.k.cov_func.warp_function(Rstar, *gp.k.params[1:]), linewidth=3)
a3.set_xlabel('$R$ [m]')
a3.set_ylabel('$l$ [m]')

"""a3 = f.add_subplot(3, 1, 3, sharex=a1)
a3.plot(Rstar, meandd, 'k', linewidth=3)
a3.plot(Rstar, meandd_approx)
a3.fill_between(Rstar, (meandd-stddd), (meandd+stddd), alpha=0.375, facecolor='k')
a3.axvline(x=R_mag_mean, color='r')
a3.axvspan(R_mag_mean-R_mag_std, R_mag_mean+R_mag_std, alpha=0.375, facecolor='r')
a3.set_xlabel('$R$ [m]')
a3.set_ylabel('$d^2T_{e}/dR^2$ [keV/m$^2$]')"""

a1.set_xlim(0.66, 0.91)
a3.set_ylim(0.0, 0.15)

a3.text(1,
        0.0,
        'C-Mod shot %d' % shot,
        rotation=90,
        transform=a3.transAxes,
        verticalalignment='bottom',
        horizontalalignment='left',
        size=12)

f.subplots_adjust(hspace=0)

# rand_vars = numpy.random.standard_normal((len(Rstar), 4))
# samps = gp.draw_sample(Rstar, rand_vars=rand_vars)
# a1.plot(Rstar, samps, linewidth=2)
# 
# deriv_samps = gp.draw_sample(Rstar, n=1, rand_vars=rand_vars, diag_factor=1e4)
# a2.plot(Rstar, deriv_samps, linewidth=2)

a2.yaxis.get_major_ticks()[-1].label.set_visible(False)
a2.yaxis.get_major_ticks()[0].label.set_visible(False)
# a3.yaxis.get_major_ticks()[-1].label.set_visible(False)
f.canvas.draw()

"""print("Computing LL matrix...this could take a while!")

sigma_f_1_bounds = (0.75, 5.0)
l_1_bounds = (0.1, 0.5)
sigma_f_2_bounds = (0.75, 5.0)
l_2_bounds = (0.001, 0.1)
num_sigma_f_1 = 24
num_l_1 = 25
num_sigma_f_2 = 26
num_l_2 = 27
ll_vals, param_vals = gptools.parallel_compute_ll_matrix(
    gp,
    [sigma_f_1_bounds, l_1_bounds, sigma_f_2_bounds, l_2_bounds],
    [num_sigma_f_1, num_l_1, num_sigma_f_2, num_l_2],
    num_proc=None
)
#ll_vals, param_vals = gp.compute_ll_matrix([sigma_f_bounds, l_bounds], [num_sigma_f, num_l])
sigma_f_1_vals = param_vals[0]
l_1_vals = param_vals[1]
sigma_f_2_vals = param_vals[2]
l_2_vals = param_vals[3]

gptools.slice_plot(ll_vals, sigma_f_1_vals, l_1_vals, sigma_f_2_vals, l_2_vals, names=['sigma f 1', 'l 1', 'sigma f 2', 'l2'])"""

#L, SIGMA_F = scipy.meshgrid(l_vals, sigma_f_vals)

"""grid = scipy.mgrid[0:num_sigma_f, 0:num_l, 0:num_sigma_n]
SIGMA_F = sigma_f_vals[grid[0]]
L = l_vals[grid[1]]
SIGMA_N = sigma_n_vals[grid[2]]

cs = mlab.contour3d(SIGMA_F/sigma_f_vals.max(), L/l_vals.max(), SIGMA_N/sigma_n_vals.max(), ll_vals, contours=200, opacity=0.5)
ax = mlab.axes(xlabel='sigma f', ylabel='l', zlabel='sigma n')
cb = mlab.colorbar(title='ll')
mlab.show()"""

"""f2 = plt.figure()
a2 = f2.add_subplot(1, 1, 1)
cs = a2.contour(L, SIGMA_F, ll_vals, 300)#, vmin=-40, vmax=0)
cbar = f2.colorbar(cs)
cbar.set_label('$\ln\, p(T\,|\,R,\, l,\, \sigma_f)$', labelpad=12)
#cbar.set_ticks(range(-45, 5, 5))
#a2.plot(gp.k.params[1], gp.k.params[0], 'k+', markersize=12)
a2.set_xlabel('$l$ [m]')
a2.set_ylabel('$\sigma_{f}$ [keV]')

a2.set_title('Contours of $\ln\, p(T\,|\,R,\, l,\, \sigma_f)$')"""
