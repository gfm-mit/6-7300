import numpy as np

from domain_specific.evalf import get_exports


def finiteDifferenceJacobian(func, x, p, u, delta = 1e-6):
    t = None
    x = x.reshape(-1,)
    F0 = func(x, t, p, u)

    J = np.zeros((F0.shape[0], x.shape[0])).astype(np.float64)

    for k in range(x.shape[0]):
        eps = np.zeros(x.shape[0])
        eps[k] = delta
        Fk = func(x + eps, t, p, u)
        J[:, k] = (Fk - F0)/delta

    return J


def evalJacobian(x, p, u):

    # First n are Yi's
    # Next n are Yi tilde
    # Last n are mu_i
    x = x.flatten()

    J = np.zeros((x.shape[0], x.shape[0])).astype(np.float64)
    n = int(x.shape[0]/3)
    n_y = 0
    n_tilde = n
    n_mu = 2 * n

    # everything but exports and imports
    for c in range(n):
        # mean reversion terms
        J[n_y + c][n_y + c] = -1/p['tau3']
        J[n_tilde + c][n_tilde + c] = -1/p['tau1']
        J[n_mu + c][n_mu + c] = -1/p['tau2']

        J[n_y + c][n_mu + c] = 1
        J[n_tilde + c][n_y + c] = 1/p['tau1']

    _, y_tilde, _ = x.reshape(3, n)
    x_xm = get_exports(y_tilde, p)

    # exports and imports
    for c_in in range(n):
        for c_out in range(n):
            if c_in == c_out:
                every_c = np.s_[:] # _magic_
                exports = -p['gamma2'][every_c] * x_xm[c_out, every_c]
                imports = -p['gamma2'][c_out] * x_xm[every_c, c_out]
            if c_in != c_out:
                exports = p['gamma2'][c_in] * x_xm[c_out, c_in]
                imports = p['gamma2'][c_out] * x_xm[c_in, c_out]
            J[n_mu + c_out][n_tilde + c_in] = np.sum(exports + imports) * p['alpha'] / p['tau2']
    return J