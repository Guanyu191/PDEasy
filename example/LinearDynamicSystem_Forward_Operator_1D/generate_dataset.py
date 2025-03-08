import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from scipy.integrate import solve_ivp


def grf_1d(domain=(0, 1), n_func=100, n_sensor=100, length_scale=0.2):

    x_sensor = np.linspace(domain[0], domain[1], n_sensor).reshape(-1, 1)

    pairwise_sq_dists = cdist(x_sensor, x_sensor, metric='euclidean')
    K = np.exp(-pairwise_sq_dists ** 2 / (2 * length_scale ** 2))

    F = np.random.multivariate_normal(
        mean=np.zeros(n_sensor), cov=K, size=n_func
    )
    # F.shape is (n_func, n_sensor)

    x_sensor = x_sensor.flatten()
    return x_sensor, F

def solve_ode_with_grf_1d(x_sensor, F, g, domain=(0, 1), n_loc=100):

    # x_sensor: grid points for generating function F (from GRF)
    # x: grid points for solving equation (interpolation for RK45)
    # X: random location for output function U

    if x_sensor.ndim > 1:
        x_sensor = x_sensor.flatten()
    n_func, _ = F.shape

    x = np.linspace(domain[0], domain[1], 5*n_loc)
    X = np.zeros((n_func, n_loc))  # (n_func, n_loc) location
    U = np.zeros((n_func, n_loc))  # (n_func, n_loc) solution

    for i in range(n_func):
        f = F[i]

        f_interp = interp1d(x_sensor, f, kind='cubic', fill_value="extrapolate")
        def g_wrapped(x, u):
            return g(x, u, f_interp(x))
        
        solution = solve_ivp(g_wrapped, domain, [0], t_eval=x, method='RK45')
        idx = np.random.choice(len(x), size=n_loc, replace=False)
        idx = np.sort(idx)
        X[i] = solution.t[idx]
        U[i] = solution.y[0][idx]

    return X, U


def generate_dataset(
        g, domain=(0, 1), n_func=100, n_sensor=100, length_scale=0.2, n_loc=100
    ):

    x_sensor, _F = grf_1d(domain, n_func, n_sensor, length_scale)
    _X, _U = solve_ode_with_grf_1d(x_sensor, _F, g, domain, n_loc)
    
    # F.shape = (n_func * n_loc, n_sensor)
    # X.shape = (n_func * n_loc, n_dim=1)
    # U.shape = (n_func * n_loc, n_dim=1)
    # refer to: https://arxiv.org/pdf/2103.10974

    F = []
    X = []
    U = []

    for i in range(n_func):
        F.append(np.repeat(_F[[i], :], n_loc, axis=0))
        X.append(_X[[i], :].reshape(-1, 1))
        U.append(_U[[i], :].reshape(-1, 1))

    F = np.concatenate(F, axis=0)
    X = np.concatenate(X, axis=0)
    U = np.concatenate(U, axis=0)
    return F, X, U


if __name__ == '__main__':
    import os
    from scipy.io import savemat

    DOMAIN = (0, 1)
    N_FUNC = 100
    N_SENSOR = 100
    N_LOC = 100
    LENGTH_SCALE = 0.2
    DATA_DIR = './data'

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    def g(x, u, f):
        return f
    
    # Trainset
    F, X, U = generate_dataset(
        g, DOMAIN, N_FUNC, N_SENSOR, LENGTH_SCALE, N_LOC
    )
    savemat(
        os.path.join(DATA_DIR, 'LinearDynamicSystem_Sol_Trainset.mat'),
        {'F': F, 'X': X, 'U': U}
    )

    # Testset
    F, X, U = generate_dataset(
        g, DOMAIN, 10*N_FUNC, N_SENSOR, LENGTH_SCALE, N_LOC
    )
    savemat(
        os.path.join(DATA_DIR, 'LinearDynamicSystem_Sol_Testset.mat'),
        {'F': F, 'X': X, 'U': U}
    )