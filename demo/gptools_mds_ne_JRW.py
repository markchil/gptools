from __future__ import division
import gptools
import eqtools
import MDSplus
import time
import scipy
import scipy.stats
import scipy.io
import numpy.random
import matplotlib
matplotlib.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt
plt.ion()
#plt.close('all')

shot = 1120907028
# Start and end times of flat top:
flat_start = 1.0
flat_stop = 1.1

efit_tree = eqtools.CModEFITTree(shot)
t_EFIT = efit_tree.getTimeBase()

electrons = MDSplus.Tree('electrons', shot)

# Get core TS data:
N_ne_TS = electrons.getNode(r'\electrons::top.yag_new.results.profiles:ne_rz')

t_ne_TS = N_ne_TS.dim_of().data()

# Only keep points that are in the RF flat top:
ok_idxs = (t_ne_TS >= flat_start) & (t_ne_TS <= flat_stop)
t_ne_TS = t_ne_TS[ok_idxs]

ne_TS = N_ne_TS.data()[:, ok_idxs] / 1e20
dev_ne_TS = electrons.getNode(r'yag_new.results.profiles:ne_err').data()[:, ok_idxs] / 1e20

Z_CTS = electrons.getNode(r'yag_new.results.profiles:z_sorted').data()
R_CTS = electrons.getNode(r'yag.results.param:r').data() * scipy.ones_like(Z_CTS)
t_grid, Z_grid = scipy.meshgrid(t_ne_TS, Z_CTS)
t_grid, R_grid = scipy.meshgrid(t_ne_TS, R_CTS)
R_mid_CTS = efit_tree.rz2rmid(R_grid, Z_grid, t_grid, each_t=False)

# Get edge data:
N_ne_ETS = electrons.getNode(r'yag_edgets.results:ne')

t_ne_ETS = N_ne_ETS.dim_of().data()
# Assume ETS is on same timebase as CTS and use indices from above:
t_ne_ETS = t_ne_ETS[ok_idxs]

ne_ETS = N_ne_ETS.data()[:, ok_idxs] / 1e20
dev_ne_ETS = electrons.getNode(r'yag_edgets.results:ne:error').data()[:, ok_idxs] / 1e20

Z_ETS = electrons.getNode(r'yag_edgets.data:fiber_z').data()
R_ETS = electrons.getNode(r'yag.results.param:R').data() * scipy.ones_like(Z_ETS)
t_grid, Z_grid = scipy.meshgrid(t_ne_ETS, Z_ETS)
t_grid, R_grid = scipy.meshgrid(t_ne_ETS, R_ETS)
R_mid_ETS = efit_tree.rz2rmid(R_grid, Z_grid, t_grid, each_t=False)

# Flag bad points for exclusion:
ne_ETS[(ne_ETS == 0) & (dev_ne_ETS == 2)] = scipy.nan

# Get magnetic axis location:
ok_idxs = (t_EFIT >= flat_start) & (t_EFIT <= flat_stop)
R_mag = efit_tree.getMagR()
R_mag_TS = scipy.interpolate.InterpolatedUnivariateSpline(t_EFIT, R_mag)(t_ne_TS)
R_mag_mean = scipy.mean(R_mag[ok_idxs])
R_mag_std = scipy.std(R_mag[ok_idxs])

# Get LCFS outboard midplane location:
R_out = efit_tree.getRmidOut()
R_out_TS = scipy.interpolate.InterpolatedUnivariateSpline(t_EFIT, R_out)(t_ne_TS)
R_out_mean = scipy.mean(R_out[ok_idxs])
R_out_std = scipy.std(R_out[ok_idxs])

# Compute weighted mean and weighted corected sample standard deviation:
# # Single time slice:
# idx = 15
# ne_TS_w = ne_TS[:, idx]
# dev_ne_TS_w = dev_ne_TS[:, idx]
# R_mid_w = R_mid_CTS[:, idx]
# dev_R_mid_w = scipy.zeros_like(R_mid_w)
# 
# ne_ETS_w = ne_ETS[:, idx]
# good_idxs = ~scipy.isnan(ne_ETS_w)
# ne_ETS_w = ne_ETS_w[good_idxs]
# dev_ne_ETS_w = dev_ne_ETS[:, idx]
# dev_ne_ETS_w = dev_ne_ETS_w[good_idxs]
# R_mid_ETS_w = R_mid_ETS[:, idx]
# R_mid_ETS_w = R_mid_ETS_w[good_idxs]
# dev_R_mid_ETS_w = scipy.zeros_like(R_mid_ETS_w)
# 
# R_mag_mean = R_mag[idx]
# R_mag_std = 0.0
# R_out_mean = R_out[idx]
# R_out_std = 0.0

# # Average over entire data set:
# ne_TS_w = scipy.mean(ne_TS, axis=1)
# dev_ne_TS_w = scipy.std(ne_TS, axis=1, ddof=1)
# R_mid_w = scipy.mean(R_mid_CTS, axis=1)
# dev_R_mid_w = scipy.std(R_mid_CTS, axis=1, ddof=1)
# 
# ne_ETS_w = scipy.stats.nanmean(ne_ETS, axis=1)
# dev_ne_ETS_w = scipy.stats.nanstd(ne_ETS, axis=1)
# R_mid_ETS_w = scipy.mean(R_mid_ETS, axis=1)
# dev_R_mid_ETS_w = scipy.std(R_mid_ETS, axis=1, ddof=1)

# Average over entire data set, try using roubust estimators:
# IQR_to_std = 1.349

robust = True
ne_TS_w, dev_ne_TS_w = gptools.compute_stats(ne_TS, robust=robust)
R_mid_w, dev_R_mid_w = gptools.compute_stats(R_mid_CTS, robust=robust)

ne_ETS_w, dev_ne_ETS_w = gptools.compute_stats(ne_ETS, robust=robust, check_nan=True)
R_mid_ETS_w, dev_R_mid_ETS_w = gptools.compute_stats(R_mid_ETS, robust=robust)

# # Make Q-Q plots with the robust statistics dictating the distribution:
# for k in xrange(0, ne_TS.shape[0]):
#     ne_ch = ne_TS[k, :]
#     ne_ch = ne_ch[~scipy.isnan(ne_ch)]
#     f = plt.figure()
#     scipy.stats.probplot(ne_ch, sparams=(ne_TS_w[k], dev_ne_TS_w[k]), plot=plt)
#     f.suptitle('CTS: idx=%d, R=%.3fm' % (k, R_mid_w[k]))
# for k in xrange(0, ne_ETS.shape[0]):
#     ne_ch = ne_ETS[k, :]
#     ne_ch = ne_ch[~scipy.isnan(ne_ch)]
#     f = plt.figure()
#     scipy.stats.probplot(ne_ch, sparams=(ne_ETS_w[k], dev_ne_ETS_w[k]), plot=plt)
#     f.suptitle('ETS: idx=%d, R=%.3fm' % (k, R_mid_ETS_w[k]))

# # Use entire data set, taking every skip-th point:
# skip = 1
# R_mid_w = R_mid_CTS.flatten()[::skip]
# ne_TS_w = ne_TS.flatten()[::skip]
# dev_ne_TS_w = dev_ne_TS.flatten()[::skip]
# dev_ne_TS_w = scipy.sqrt((dev_ne_TS_w)**2.0 +
#                          (scipy.repeat(scipy.std(ne_TS, axis=1), ne_TS.shape[1])[::skip])**2)
# 
# ne_ETS_w = ne_ETS.flatten()[::skip]
# good_idxs = ~scipy.isnan(ne_ETS_w)
# ne_ETS_w = ne_ETS_w[good_idxs]
# R_mid_ETS_w = R_mid_ETS.flatten()[::skip]
# R_mid_ETS_w = R_mid_ETS_w[good_idxs]
# dev_ne_ETS_w = dev_ne_ETS.flatten()[::skip]
# dev_ne_ETS_w = dev_ne_ETS_w[good_idxs]
# dev_ne_ETS_w = scipy.sqrt(((dev_ne_ETS).flatten()[::skip])**2.0 +
#                           (scipy.repeat(scipy.std(ne_ETS, axis=1), ne_ETS.shape[1])[::skip])**2)

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
    initial_params=[0.1, 0.1, 0.05, 0.1, 0.89],
    fixed_params=[False, False, False, False, False],
    param_bounds=[(0.0, 10000.0), (0.01, 10.0), (0.0001, 1.0), (0.0001, 10.0), (0.88, 0.91)],
    enforce_bounds=True
)
# k = gptools.GibbsKernel1dQuinticBucket(initial_params=[1, 0.15, 0.15, 0.1, 0.75, 0.1, 0.1, 0.1],
#                                        fixed_params=[False, False, False, False, False, False, False, False],
#                                        param_bounds=[(0.0, 1000.0),  # sigmaf
#                                                      (0.01, 10.0),   # l1
#                                                      (0.01, 10.0),   # l2
#                                                      (0.001, 10.0),  # l3
#                                                      (0.7, 0.95),    # x0
#                                                      (0.0001, 0.1),  # w1
#                                                      (0.0001, 1.0),  # w2 (bucket width)
#                                                      (0.0001, 0.1)], # w3
#                                        enforce_bounds=True)

# Set noise kernel:
nk = gptools.DiagonalNoiseKernel(1, n=0, initial_noise=0.0, fixed_noise=True, noise_bound=(0.0, 10.0))

# Create and populate GP:
gp = gptools.GaussianProcess(k, noise_k=nk)
gp.add_data(R_mid_w, ne_TS_w, err_y=dev_ne_TS_w)
gp.add_data(R_mid_ETS_w, ne_ETS_w, err_y=dev_ne_ETS_w)
gp.add_data(R_mag_mean, 0, n=1)
# gp.add_data(0.904, 0, err_y=0.1)
# gp.add_data(0.904, 0, n=1, err_y=1)
# gp.add_data(0.91, 0, err_y=0.01)
# gp.add_data(0.91, 0, n=1, err_y=0.1)
# gp.add_data(0.95, 0, err_y=0.001)
# gp.add_data(0.95, 0, n=1, err_y=0.1)

# Try block constraint:
R_out = scipy.linspace(0.92, 0.95, 5)
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
            # {'type': 'ineq', 'fun': gptools.Constraint(
            #     gp,
            #     n=1,
            #     type_='lt',
            #     boundary_val=0.1,
            #     loc=0.902,
            # )},
            # {'type': 'ineq', 'fun': gptools.Constraint(
            #     gp,
            #     n=1,
            #     type_='lt',
            #     boundary_val=0.1,
            #     loc=0.93,
            # )},
            # {'type': 'ineq', 'fun': gptools.Constraint(
            #     gp,
            #     n=0,
            #     type_='lt',
            #     boundary_val=0.02,
            #     loc=0.909,
            # )},
            # {'type': 'ineq', 'fun': gptools.Constraint(
            #     gp,
            #     n=0,
            #     type_='gt',
            #     boundary_val=0.0,
            #     loc=0.909,
            # )},
            # {'type': 'ineq', 'fun': gptools.Constraint(
            #     gp,
            #     n=0,
            #     type_='lt',
            #     boundary_val=0.02,
            #     loc=0.912,
            # )},
            # {'type': 'ineq', 'fun': gptools.Constraint(
            #     gp,
            #     n=0,
            #     type_='gt',
            #     boundary_val=0.0,
            #     loc=0.912,
            # )},
            # {'type': 'ineq', 'fun': gptools.Constraint(
            #     gp,
            #     n=0,
            #     type_='gt',
            #     boundary_val=0.0,
            #     loc=0.902,
            # )},
            # {'type': 'ineq', 'fun': gptools.Constraint(
            #     gp,
            #     n=0,
            #     type_='gt',
            #     boundary_val=0.0,
            #     loc=0.93,
            # )},
            # {'type': 'ineq', 'fun': gptools.Constraint(gp)},
            # {'type': 'ineq', 'fun': lambda p: p[1] - p[2]},
        )
    }
)
opt_elapsed = time.time() - opt_start

# Make predictions:
# Rstar = scipy.linspace(0.63, 0.93, 24*30)
# Get Rstar from a fits savefile:
fits_file = scipy.io.readsav('/home/markchil/codes/gptools/demo/nth_samples_1101014006.save')
Rstar = fits_file.ne_fit.rmajor[0][:, 0]
ne_nth = fits_file.ne_fit.combined_fit_ne[0][:, 32:72] / 1e20
mean_nth, std_nth = gptools.compute_stats(ne_nth, robust=robust)

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

f.suptitle('Univariate GPR on TS data')
# f.suptitle('Univariate GPR on time-averaged data')
# f.suptitle('Univariate GPR on single frame of TS data, $t=%.2fs$' % t_ne_TS[idx])
# f.suptitle('With slope constraint')

a1 = f.add_subplot(3, 1, 1)
a1.plot(Rstar, mean, 'k', linewidth=3, label='GPR')
a1.fill_between(Rstar, mean-std, mean+std, alpha=0.375, facecolor='k')
a1.plot(Rstar, mean_nth, 'g', linewidth=3, label='spline')
a1.fill_between(Rstar, mean_nth-std_nth, mean_nth+std_nth, alpha=0.375, facecolor='g')
# a1.plot(Rstar, ne_nth)
a1.errorbar(R_mid_w, ne_TS_w, xerr=dev_R_mid_w, yerr=dev_ne_TS_w, fmt='r.', label='CTS') # 
a1.errorbar(R_mid_ETS_w, ne_ETS_w, xerr=dev_R_mid_ETS_w, yerr=dev_ne_ETS_w, fmt='m.', label='ETS') # 
a1.axvline(x=R_mag_mean, color='r')#, label='$R_{mag}$')
a1.axvspan(R_mag_mean-R_mag_std, R_mag_mean+R_mag_std, alpha=0.375, facecolor='r')
# a1.axvline(x=R_out_mean, color='g', label='$R_{out}$')
# a1.axvspan(R_out_mean-R_out_std, R_out_mean+R_out_std, alpha=0.375, facecolor='g')
a1.legend(loc='upper right', fontsize=10, ncol=2)
#a1.set_xlabel('$R$ [m]')
a1.get_xaxis().set_visible(False)
a1.set_ylim(bottom=0)
a1.set_ylabel('$n_{e}$ [$10^{20}$m$^{-3}$]')

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
a2.set_ylabel('$dn_{e}/dR$\n[$10^{20}$m$^{-4}$]')

a3 = f.add_subplot(3, 1, 3, sharex=a1)
a3.plot(Rstar, gp.k.l_func(Rstar, 0, *gp.k.params[1:]), linewidth=3)
a3.set_xlabel('$R$ [m]')
a3.set_ylabel('$l$ [m]')

a1.set_xlim(0.67, 0.91)
a3.set_ylim(bottom=0.0)

a3.text(1,
        0.0,
        'C-Mod shot %d' % shot,
        rotation=90,
        transform=a3.transAxes,
        verticalalignment='bottom',
        horizontalalignment='left',
        size=12)

f.subplots_adjust(hspace=0)

# rand_vars = numpy.random.standard_normal((len(Rstar), 10))
# samps = gp.draw_sample(Rstar, rand_vars=rand_vars, method='eig', num_eig=10)
# a1.plot(Rstar, samps, linewidth=2)
# 
# deriv_samps = gp.draw_sample(Rstar, n=1, rand_vars=rand_vars, diag_factor=1e4, method='eig', num_eig=10)
# a2.plot(Rstar, deriv_samps, linewidth=2)

samp_loc = Rstar
samp_n = scipy.zeros_like(Rstar)
num_samp = 99
num_eig = 10

loc_arr = scipy.reshape(scipy.repeat(samp_loc, num_samp), (len(samp_loc), -1))
rand_vars = numpy.random.standard_normal((num_eig, num_samp))
samps = gp.draw_sample(samp_loc, n=samp_n, rand_vars=rand_vars, method='eig', num_eig=num_eig)

a2.yaxis.get_major_ticks()[-1].label.set_visible(False)
a2.yaxis.get_major_ticks()[0].label.set_visible(False)
# a3.yaxis.get_major_ticks()[-1].label.set_visible(False)
f.canvas.draw()

with open('ne.dat', 'wb') as nefile:
    nefile.write(scipy.array(1e20 * samps, dtype=scipy.float32))
# Assume Rmaj.dat was written by Te program.
