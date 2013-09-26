import gptools
import time
import cPickle as pickle
import scipy
import scipy.stats
import numpy.random
import matplotlib
matplotlib.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt
plt.ion()
#plt.close('all')

#import mayavi.mlab as mlab

with open('data.p', 'rb') as pf:
    data_dict = pickle.load(pf)

Z_CTS = data_dict['Z']
Z_sort = Z_CTS.argsort()
Z_CTS = Z_CTS[Z_sort]
Te_TS = data_dict['Te'][Z_sort, :]
Te_TS = scipy.delete(Te_TS, (10,), axis=0)
dev_Te_TS = data_dict['dev_Te'][Z_sort, :]
dev_Te_TS = scipy.delete(dev_Te_TS, (10,), axis=0)
t_Te_TS = data_dict['t']
R_mid_CTS = data_dict['R_mid'][Z_sort, :]
R_mid_CTS = scipy.delete(R_mid_CTS, (10,), axis=0)
R_mag = data_dict['R_mag']

Te_ETS = data_dict['Te_ETS'] / 1e3
t_Te_ETS = data_dict['t_ETS']
dev_Te_ETS = data_dict['dev_Te_ETS'] / 1e3
R_mid_ETS = data_dict['R_mid_ETS']

Te_ETS[(Te_ETS == 0) & (dev_Te_ETS == 1)] = scipy.nan

R_mag_mean = scipy.mean(R_mag)
R_mag_std = scipy.std(R_mag)

# Compute weighted mean and weighted corected sample standard deviation:
idx = 44
Te_TS_w = Te_TS[:, idx]
dev_Te_TS_w = dev_Te_TS[:, idx]
R_mid_w = R_mid_CTS[:, idx]

Te_ETS_w = Te_ETS[:, idx]
good_idxs = ~scipy.isnan(Te_ETS_w)
Te_ETS_w = Te_ETS_w[good_idxs]
dev_Te_ETS_w = dev_Te_ETS[:, idx]
dev_Te_ETS_w = dev_Te_ETS_w[good_idxs]
R_mid_ETS_w = R_mid_ETS[:, idx]
R_mid_ETS_w = R_mid_ETS_w[good_idxs]

# Te_TS_w = scipy.mean(Te_TS, axis=1)
# dev_Te_TS_w = scipy.std(Te_TS, axis=1)
# R_mid_w = scipy.mean(R_mid_CTS, axis=1)
# dev_R_mid_w = scipy.std(R_mid_CTS, axis=1)

# Te_ETS_w = scipy.stats.nanmean(Te_ETS, axis=1)
# dev_Te_ETS_w = scipy.stats.nanstd(Te_ETS, axis=1)
# R_mid_ETS_w = scipy.mean(R_mid_ETS, axis=1)
# dev_R_mid_ETS_w = scipy.std(R_mid_ETS, axis=1)

# skip = 1
# R_mid_w = R_mid_CTS.flatten()[::skip]
# Te_TS_w = Te_TS.flatten()[::skip]
# dev_Te_TS_w = dev_Te_TS.flatten()[::skip]
# dev_Te_TS_w = 4*scipy.sqrt((dev_Te_TS_w)**2.0 +
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

k = gptools.SquaredExponentialKernel(1,
                                     initial_params=[1, 0.15],
                                     fixed_params=[False, False],
                                     param_bounds=[(0.0, 1000.0), (0.01, 1.0)])
# k = gptools.MaternKernel(1,
#                          initial_params=[1, 3.0/2.0, 0.15],
#                          fixed_params=[False, False, False],
#                          param_bounds=[(0.0, 1000.0), (0.51, 10.0), (0.01, 1.0)])
# k = gptools.RationalQuadraticKernel(1,
#                                     initial_params=[1, 20.0, 0.15],
#                                     fixed_params=[False, False, False],
#                                     param_bounds=[(0.0, 1000.0), (0.001, 100.0), (0.01, 1.0)],
#                                     enforce_bounds=True)
# k = gptools.GibbsKernel1dtanh(initial_params=[1, 0.15, 0.01, 0.005, 0.89],
#                               fixed_params=[False, False, False, False, False],
#                               param_bounds=[(0.0, 1000.0), (0.01, 10.0), (0.0001, 1.0), (0.0001, 0.1), (0.88, 0.91)],
#                               num_proc=None,
#                               enforce_bounds=True)

nk = gptools.DiagonalNoiseKernel(1, n=0, initial_noise=0.0, fixed_noise=True, noise_bound=(0.0001, 10.0))
"""nk = (gptools.DiagonalNoiseKernel(1, n=0, initial_noise=0.1, fixed_noise=False) +
      gptools.DiagonalNoiseKernel(1, n=1, initial_noise=0.0, fixed_noise=True) +
      gptools.SquaredExponentialKernel(1, initial_params=[1, 0.01], fixed_params=[False, False]))"""

gp = gptools.GaussianProcess(k, noise_k=nk, X=R_mid_w, y=Te_TS_w, err_y=dev_Te_TS_w)
# gp.add_data(R_mid_ETS_w, Te_ETS_w, err_y=dev_Te_ETS_w)
gp.add_data(R_mag_mean, 0, n=1)
#gp.add_data(R_mag_mean, 0, n=2)
#gp.add_data(R_mag_mean, 0, n=3)
#gp.add_data(0.95, 0, n=0)
#gp.add_data(0.95, 0, n=1)

def l_cf(params):
    return params[1] - params[2]

class pos_cf(object):
    def __init__(self, idx):
        self.idx = idx
    def __call__(self, params):
        return params[self.idx]

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
            {'type': 'ineq', 'fun': l_cf},
            # {'type': 'ineq', 'fun': gptools.Constraint(gp, n=1, type_='lt', loc='max')},
            #{'type': 'ineq', 'fun': gptools.Constraint(gp)},
        )
    }
)
opt_elapsed = time.time() - opt_start

Rstar = scipy.linspace(R_mag_mean, R_mid_ETS_w.max(), 24*100)

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
f.suptitle('Univariate GPR on single frame of TS data, $t=%.2fs$' % t_Te_TS[idx])
#f.suptitle('With slope constraint')

a1 = f.add_subplot(3, 1, 1)
a1.plot(Rstar, mean, 'k', linewidth=3)
a1.fill_between(Rstar, mean-std, mean+std, alpha=0.375, facecolor='k')
a1.errorbar(R_mid_w, Te_TS_w, yerr=dev_Te_TS_w, fmt='r.')  #, xerr=dev_R_mid_w
a1.errorbar(R_mid_ETS_w, Te_ETS_w, yerr=dev_Te_ETS_w, fmt='m.')  #, xerr=dev_R_mid_ETS_w
a1.axvline(x=R_mag_mean, color='r')
a1.axvspan(R_mag_mean-R_mag_std, R_mag_mean+R_mag_std, alpha=0.375, facecolor='r')
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

a1.set_xlim(0.69, 0.92)
a3.set_ylim(0.0, 0.1)

a3.text(1,
        0.0,
        'C-Mod shot %d' % data_dict['shot'],
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
