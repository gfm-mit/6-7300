from evalf import evalf, get_E, generate_inputs
import numpy as np


def finiteDifferenceJacobian(func, x, p, u, delta = 1e-6):
    F0 = func(x, p, u)

    J = np.zeros((F0.shape[0], x.shape[0])).astype(np.float64)

    for k in range(x.shape[0]):
        eps = np.zeros(x.shape[0])
        eps[k] = delta
        Fk = func(x + eps, p, u)
        J[:,k] = (Fk - F0)/delta

    return J


def evalJacobian(x, p, u):

    # First n are Yi's
    # Next n are Yi tilde
    # Last n are mu_i

    J = np.zeros((x.shape[0], x.shape[0])).astype(np.float64)
    n = int(x.shape[0]/3)

    for i in range(n):
        for j in range(n):
            if i == j:
                J[i][j] = x[2*n + i] + p['sigma'][i] * u[i]  # dF(Y_i) / dYi
                J[i][2 * n + j] = x[i] # dF(Y_i) / dmu_i

                J[n + i][j] = 1/p['tau1'][i]   # dF(Yt_i) / dYi
                J[n + i][n + j] = -1/p['tau1'][i]   # dF(Yt_i) / dYt_i

                J[2 * n + i][2 * n + j] = -1/p['tau2'][i]  # dF(mu_i) / dmu_i

                sum1 = 0
                sum2 = 0

                for k in range(n):
                    if k == i: continue
                    sum1 += p['g'][k] * (x[n + k] ** p['gamma2'])/p['d'][i][k]
                    sum2 += p['g'][k] / (p['d'][k][i] * (x[n + k] ** p['gamma2']))

                J[2 * n + i][n + j] = -p['alpha'] * p['g'][i] * p['gamma2']/(p['tau2'][i] * p['gw']) * ( sum1/(x[n + i] ** (1 + p['gamma2'])) + sum2 * (x[n + i] ** (p['gamma2'] - 1)) )  # dF(mu_i) / dYt_i

            else:
                J[i][j] = 0 # dF(Y_i) / dYj
                J[i][2 * n + j] = 0 # dF(Y_i) / dmu_j

                J[n + i][j] = 0   # dF(Yt_i) / dYj
                J[n + i][j] = 0   # dF(Yt_i) / dYt_j

                J[2 * n + i][2 * n + j] = 0  # dF(mu_i) / dmu_j

                J[2 * n + i][n + j] = p['alpha'] * p['g'][i] * p['gamma2'] * p['g'][j]/(p['tau2'][i] * p['gw'] * p['d'][i][j])
                J[2 * n + i][n + j] *= ( (x[n + j] ** (p['gamma2']-1))/(x[n + i] ** p['gamma2']) + (x[n + i] ** p['gamma2'])/(x[n + j] ** (p['gamma2']+1)))  # dF(mu_i) / dYt_j

            J[i][n + j] = 0 # dF(Y_i) / dYt_j
            J[n + i][2 * n + j] = 0 # dF(Yt_i) / dmu_j
            J[2 * n + i][j] = 0   # dF(mu_i) / dY_j


if __name__ == '__main__':
    n = 3
    E = get_E('configs/test.txt')
    # t = np.linspace(0, 10, 10)
    x, p, u = generate_inputs(n, E)
    finiteDifferenceJacobian(evalf, x, p, u)
    # evalf(x, t, p, u, E)
