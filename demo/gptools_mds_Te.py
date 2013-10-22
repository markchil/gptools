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
flat_start = 0.965#0.5
flat_stop = 1.365#1.5

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

# Flag bad points for exclusion:
Te_GPC2[(Te_GPC2 == 0.0)] = scipy.nan

# Get GPC data:
Te_GPC = []
R_mid_GPC = []
for k in xrange(0, 9):
    N = electrons.getNode(r'ece.gpc_results.te.te%d' % (k + 1,))
    Te_GPC.append(N.data())
    N_R = electrons.getNode(r'ece.gpc_results.rad.r%d' % (k + 1,))
    R_mid_GPC.append(N_R.data())
# Assume all slow channels are on the same timebase:
t_GPC = N.dim_of().data()
# The radius is given on the EFIT timebase, so must be handled separately:
t_R_GPC = N_R.dim_of().data()
# Remove points outside of range of interest:
ok_idxs = (t_GPC >= flat_start) & (t_GPC <= flat_stop)
t_GPC = t_GPC[ok_idxs]
Te_GPC = scipy.asarray(Te_GPC, dtype=float)[:, ok_idxs]
# Handling the radius like this will be fine for shot-average, but needs to be
# interpolated to handle single-frame:
ok_idxs = (t_R_GPC >= flat_start) & (t_R_GPC <= flat_stop)
t_R_GPC = t_R_GPC[ok_idxs]
R_mid_GPC = scipy.asarray(R_mid_GPC, dtype=float)[:, ok_idxs]

# Get magnetic axis location:
ok_idxs = (t_EFIT >= flat_start) & (t_EFIT <= flat_stop)
R_mag = efit_tree.getMagR()
R_mag_TS = scipy.interpolate.InterpolatedUnivariateSpline(t_EFIT, R_mag)(t_Te_TS)
R_mag_mean = scipy.mean(R_mag[ok_idxs])
R_mag_std = scipy.std(R_mag[ok_idxs])

# Get LCFS outboard midplane location:
R_out = efit_tree.getRmidOut()
R_out_TS = scipy.interpolate.InterpolatedUnivariateSpline(t_EFIT, R_out)(t_Te_TS)
R_out_mean = scipy.mean(R_out[ok_idxs])
R_out_std = scipy.std(R_out[ok_idxs])

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

# # Average over entire data set:
# Te_TS_w = scipy.mean(Te_TS, axis=1)
# dev_Te_TS_w = scipy.std(Te_TS, axis=1, ddof=1)
# R_mid_w = scipy.mean(R_mid_CTS, axis=1)
# dev_R_mid_w = scipy.std(R_mid_CTS, axis=1, ddof=1)
# 
# Te_ETS_w = scipy.stats.nanmean(Te_ETS, axis=1)
# dev_Te_ETS_w = scipy.stats.nanstd(Te_ETS, axis=1)
# R_mid_ETS_w = scipy.mean(R_mid_ETS, axis=1)
# dev_R_mid_ETS_w = scipy.std(R_mid_ETS, axis=1, ddof=1)
# 
# Te_FRC_w = scipy.mean(Te_FRC, axis=1)
# dev_Te_FRC_w = scipy.std(Te_FRC, axis=1, ddof=1)
# R_mid_FRC_w = scipy.mean(R_mid_FRC, axis=1)
# dev_R_mid_FRC_w = scipy.std(R_mid_FRC, axis=1, ddof=1)
# # Get rid of clearly too small points (Why do these happen?)
# good_idxs = (Te_FRC_w >= 0.1)
# Te_FRC_w = Te_FRC_w[good_idxs]
# dev_Te_FRC_w = dev_Te_FRC_w[good_idxs]
# R_mid_FRC_w = R_mid_FRC_w[good_idxs]
# dev_R_mid_FRC_w = dev_R_mid_FRC_w[good_idxs]
# 
# Te_GPC2_w = scipy.stats.nanmean(Te_GPC2, axis=1)
# dev_Te_GPC2_w = scipy.stats.nanstd(Te_GPC2, axis=1)
# R_mid_GPC2_w = scipy.mean(R_mid_GPC2, axis=1)
# dev_R_mid_GPC2_w = scipy.std(R_mid_GPC2, axis=1, ddof=1)
# # Get rid of bad channels and channels outside the pedestal:
# bad_idxs = scipy.where(scipy.isnan(Te_GPC2_w) | (R_mid_GPC2_w >= 0.9))[0]
# Te_GPC2_w = scipy.delete(Te_GPC2_w, bad_idxs)
# dev_Te_GPC2_w = scipy.delete(dev_Te_GPC2_w, bad_idxs)
# R_mid_GPC2_w = scipy.delete(R_mid_GPC2_w, bad_idxs)
# dev_R_mid_GPC2_w = scipy.delete(dev_R_mid_GPC2_w, bad_idxs)
# 
# Te_GPC_w = scipy.mean(Te_GPC, axis=1)
# dev_Te_GPC_w = scipy.std(Te_GPC, axis=1, ddof=1)
# R_mid_GPC_w = scipy.mean(R_mid_GPC, axis=1)
# dev_R_mid_GPC_w = scipy.std(R_mid_GPC, axis=1, ddof=1)
# # Get rid of clearly too small points (Why do these happen?)
# good_idxs = (Te_GPC_w >= 0.1)
# Te_GPC_w = Te_GPC_w[good_idxs]
# dev_Te_GPC_w = dev_Te_GPC_w[good_idxs]
# R_mid_GPC_w = R_mid_GPC_w[good_idxs]
# dev_R_mid_GPC_w = dev_R_mid_GPC_w[good_idxs]

# Average over entire data set, use robust estimators:
IQR_to_std = 1.349

robust = True
Te_TS_w, dev_Te_TS_w = gptools.compute_stats(Te_TS, robust=robust)
R_mid_w, dev_R_mid_w = gptools.compute_stats(R_mid_CTS, robust=robust)

Te_ETS_w, dev_Te_ETS_w = gptools.compute_stats(Te_ETS, robust=robust, check_nan=True)
R_mid_ETS_w, dev_R_mid_ETS_w = gptools.compute_stats(R_mid_ETS, robust=robust)

Te_FRC_w, dev_Te_FRC_w = gptools.compute_stats(Te_FRC, robust=robust)
R_mid_FRC_w, dev_R_mid_FRC_w = gptools.compute_stats(R_mid_FRC, robust=robust)
# Get rid of clearly too small points (Why do these happen?)
good_idxs = (Te_FRC_w >= 0.1)
Te_FRC_w = Te_FRC_w[good_idxs]
dev_Te_FRC_w = dev_Te_FRC_w[good_idxs]
R_mid_FRC_w = R_mid_FRC_w[good_idxs]
dev_R_mid_FRC_w = dev_R_mid_FRC_w[good_idxs]

Te_GPC2_w, dev_Te_GPC2_w = gptools.compute_stats(Te_GPC2, robust=robust, check_nan=True)
R_mid_GPC2_w, dev_R_mid_GPC2_w = gptools.compute_stats(R_mid_GPC2, robust=robust)
# Get rid of bad channels and channels outside the pedestal:
bad_idxs = scipy.where(scipy.isnan(Te_GPC2_w) | (R_mid_GPC2_w >= 0.9) | (R_mid_GPC2_w <= R_mag_mean))[0]
Te_GPC2_w = scipy.delete(Te_GPC2_w, bad_idxs)
dev_Te_GPC2_w = scipy.delete(dev_Te_GPC2_w, bad_idxs)
R_mid_GPC2_w = scipy.delete(R_mid_GPC2_w, bad_idxs)
dev_R_mid_GPC2_w = scipy.delete(dev_R_mid_GPC2_w, bad_idxs)

Te_GPC_w, dev_Te_GPC_w = gptools.compute_stats(Te_GPC, robust=robust)
R_mid_GPC_w, dev_R_mid_GPC_w = gptools.compute_stats(R_mid_GPC, robust=robust)
# Get rid of clearly too small points (Why do these happen?)
good_idxs = (Te_GPC_w >= 0.1)
Te_GPC_w = Te_GPC_w[good_idxs]
dev_Te_GPC_w = dev_Te_GPC_w[good_idxs]
R_mid_GPC_w = R_mid_GPC_w[good_idxs]
dev_R_mid_GPC_w = dev_R_mid_GPC_w[good_idxs]

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
k = gptools.GibbsKernel1dTanh(
    initial_params=[1.88, 0.09655, 0.05637, 0.002941, 0.8937],
    fixed_params=[False, False, False, False, False],
    param_bounds=[(0.0, 1000.0), (0.01, 10.0), (0.0001, 1.0), (0.0001, 0.1), (0.88, 0.91)],
    enforce_bounds=True
)

# Set noise kernel:
nk = gptools.DiagonalNoiseKernel(1, n=0, initial_noise=0.0, fixed_noise=True, noise_bound=(0.0, 10.0))

# Create and populate GP:
gp = gptools.GaussianProcess(k, noise_k=nk)
gp.add_data(R_mid_w, Te_TS_w, err_y=dev_Te_TS_w)
gp.add_data(R_mid_ETS_w, Te_ETS_w, err_y=dev_Te_ETS_w)
# gp.add_data(R_mid_FRC_w, Te_FRC_w, err_y=dev_Te_FRC_w)
gp.add_data(R_mid_GPC2_w, Te_GPC2_w, err_y=dev_Te_GPC2_w)
gp.add_data(R_mid_GPC_w, Te_GPC_w, err_y=dev_Te_GPC_w)
gp.add_data(R_mag_mean, 0, n=1)

# Try block constraint:
R_out = scipy.linspace(0.91, 0.95, 5)
zeros_out = scipy.zeros_like(R_out)
gp.add_data(R_out, zeros_out, err_y=0.001)
gp.add_data(R_out, zeros_out, err_y=0.1, n=1)

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
Rstar = scipy.linspace(0.63, 0.93, 24*30)

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

print("Optimization took: %.2fs\nMean prediction took: %.2fs\nGradient "
      "prediction took: %.2fs\n"  % (opt_elapsed, mean_elapsed, meand_elapsed))

# meand_approx = scipy.gradient(mean, Rstar[1] - Rstar[0])

f = plt.figure()

# f.suptitle('Univariate GPR on TS data')
f.suptitle('Univariate GPR on time-averaged data')
# f.suptitle('Univariate GPR on single frame of TS data, $t=%.2fs$' % t_Te_TS[idx])
# f.suptitle('With slope constraint')

a1 = f.add_subplot(3, 1, 1)
a1.plot(Rstar, mean, 'k', linewidth=3, label='mean')
a1.fill_between(Rstar, mean-std, mean+std, alpha=0.375, facecolor='k')
a1.errorbar(R_mid_w, Te_TS_w, xerr=dev_R_mid_w, yerr=dev_Te_TS_w, fmt='r.', label='CTS')
a1.errorbar(R_mid_ETS_w, Te_ETS_w, xerr=dev_R_mid_ETS_w, yerr=dev_Te_ETS_w, fmt='m.', label='ETS')
a1.errorbar(R_mid_FRC_w, Te_FRC_w, xerr=dev_R_mid_FRC_w, yerr=dev_Te_FRC_w, fmt='b.', label='FRC')
a1.errorbar(R_mid_GPC2_w, Te_GPC2_w, xerr=dev_R_mid_GPC2_w, yerr=dev_Te_GPC2_w, fmt='g.', label='GPC2')
a1.errorbar(R_mid_GPC_w, Te_GPC_w, xerr=dev_R_mid_GPC_w, yerr=dev_Te_GPC_w, fmt='k.', label='GPC')
a1.axvline(x=R_mag_mean, color='r', label='$R_{mag}$')
a1.axvspan(R_mag_mean-R_mag_std, R_mag_mean+R_mag_std, alpha=0.375, facecolor='r')
a1.axvline(x=R_out_mean, color='g', label='$R_{out}$')
a1.axvspan(R_out_mean-R_out_std, R_out_mean+R_out_std, alpha=0.375, facecolor='g')
a1.legend(loc='best', fontsize=10, ncol=2)
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
a3.plot(Rstar, gp.k.l_func(Rstar, 0, *gp.k.params[1:]), linewidth=3)
a3.set_xlabel('$R$ [m]')
a3.set_ylabel('$l$ [m]')

a1.set_xlim(0.63, 0.93)
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
