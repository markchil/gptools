import numpy as np
import gptools

def test_matern_52():
    # This test compares MaternKernelArb and Matern52Kernel using some
    # randomly generated points with 0th and 1st derivative observations

    f_X = np.random.RandomState(0).randn(5,2)
    f_y = f_X[:,0]**2 + f_X[:,1]**2
    g_y_0 = 2*f_X[:, 0]
    g_y_1 = 2*f_X[:, 1]

    n_dims = 2
    length_scales = np.random.lognormal(size=n_dims).tolist()
    K1 = gptools.MaternKernelArb(num_dim=2,
        initial_params=[10, 5.0/2.0,] + length_scales)
    K2 = gptools.Matern52Kernel(num_dim=2,
        initial_params=[10,] + length_scales)

    gp1 = gptools.GaussianProcess(K1)
    gp2 = gptools.GaussianProcess(K2)
    

    gp1.add_data(f_X, f_y)
    gp1.add_data(f_X, g_y_0, n=np.vstack((np.ones(len(f_X)), np.zeros(len(f_X)))).T)
    gp1.add_data(f_X, g_y_1, n=np.vstack((np.zeros(len(f_X)), np.ones(len(f_X)))).T)

    k1 = gp1.compute_Kij(gp1.X, None, gp1.n, None)
    k2 = gp2.compute_Kij(gp1.X, None, gp1.n, None)

    np.testing.assert_array_almost_equal(k1, k2, decimal=8)
