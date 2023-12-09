import numpy as np


def get_exports(y_tilde, p):
    n = y_tilde.shape[0]
    # initialize derived parameter
    g_w = np.sum(p['g'])
    # initialize component quantities
    np.fill_diagonal(p['d'], 1)
    x_xm = 1 / p['d'] / g_w
    np.fill_diagonal(x_xm, 0)
    x_xm *= p['g'][:, None] * p['g'][None, :]
    elasticity = p['gamma2'][None, :] * (y_tilde[None, :] - y_tilde[:, None])
    x_xm *= np.exp(elasticity)
    np.fill_diagonal(x_xm, 0)
    assert not np.isnan(np.sum(x_xm))
    return x_xm


def get_exports_for_loops(y_tilde, p):
    n = y_tilde.shape[0]
    # initialize derived parameter
    g_w = np.sum(p['g'])
    # initialize component quantities
    x_xm = np.zeros([n, n])
    for x in range(n):
        for m in range(n):
            if x != m:
                x_xm[x, m] = p['g'][x] * p['g'][m] / g_w / p['d'][x, m]
                elasticity = np.exp(p['gamma2'][m] * (y_tilde[m] - y_tilde[x]))
                x_xm[x, m] *= elasticity
    # update node quantities
    return x_xm


def evalf(x, t, p, u, yield_intermediates=False):
    """
    Removed gamma1 and nu
    Removed y_w, P_i and P_j

    :param x (array): state vector
    [y, tilde_y, mu]
    - y = true currency
    - tilde_y = effective currency
    - mu = currency drift
    :param p (dict): parameters
    [tau1, tau2, sigma, alpha, d, v, g]
    - tau1 = delay between true and effective currency
    - tau2 = delay between trade imbalance and currency drift
    - sigma = strength of market volatility
    - alpha = converts trade imbalance into monetary value
    - d = distance
    - v = value differential parameter
    - g = GDP at each node
    :param u (array): inputs
    [delt_w]
    :return: delt_x = f(x, p, u)
    """
    assert isinstance(x, np.ndarray)
    assert t is None or isinstance(t, float)
    assert isinstance(p, dict)
    # Reshape x (had to flatten to make it work with scipy solver)
    n = x.shape[0] // 3
    y, y_tilde, mu = x.reshape(3, n)
    assert np.max(y_tilde) - np.min(y_tilde) < 1e6, "range of currency values now exceeds 1e6: {}".format(
        y_tilde.round(3)
    )

    # initialize component quantities
    x_ij = get_exports(y_tilde, p)

    # update node quantities
    delt_true_currency = mu - y / p['tau3'] # No Weiner process
    delt_eff_currency = (y - y_tilde) / p['tau1']
    exports = np.sum(x_ij, axis=1)
    imports = np.sum(x_ij, axis=0)
    N = exports - imports
    delt_currency_drift = (p['alpha'] * N - mu) / p['tau2']

    # Flatten X (to make compatible with scipy solver)
    x_dot = np.concatenate([
        delt_true_currency,
        delt_eff_currency,
        delt_currency_drift,
        ])
    if yield_intermediates:
        return x_dot, N
    return x_dot


def evalf_for_loops(x, t, p, u):
    """
    Removed gamma1 and nu
    Removed y_w, P_i and P_j

    :param x (array): state vector
    [y, tilde_y, mu]
    - y = true currency
    - tilde_y = effective currency
    - mu = currency drift
    :param p (dict): parameters
    [tau1, tau2, sigma, alpha, d, v, g]
    - tau1 = delay between true and effective currency
    - tau2 = delay between trade imbalance and currency drift
    - sigma = strength of market volatility
    - alpha = converts trade imbalance into monetary value
    - d = distance
    - v = value differential parameter
    - g = GDP at each node
    :param u (array): inputs
    [delt_w]
    :return: delt_x = f(x, p, u)
    """
    assert isinstance(x, np.ndarray)
    assert t is None or isinstance(t, float)
    assert isinstance(p, dict)
    # Reshape x (had to flatten to make it work with scipy solver)
    n = x.shape[0] // 3
    y, y_tilde, mu = x.reshape(3, n)
    assert np.max(y_tilde) - np.min(y_tilde) < 1e6, "range of currency values now exceeds 1e6: {}".format(
        y_tilde.round(3)
    )

    # initialize node quantities
    delt_true_currency = np.zeros([n])
    delt_eff_currency = np.zeros([n])
    N = np.zeros([n])
    delt_currency_drift = np.zeros([n])

    # initialize component quantities
    x_ij = get_exports(y_tilde, p)

    # update node quantities
    for i in range(n):
        delt_true_currency[i] = mu[i] - y[i] / p['tau3']
        delt_eff_currency[i] = (y[i] - y_tilde[i]) / p['tau1']
        exports = x_ij[i]
        imports = x_ij[:, i]
        N[i] = np.sum(exports) - np.sum(imports)
        delt_currency_drift[i] = (p['alpha'] * N[i] - mu[i]) / p['tau2']

    # Flatten X (to make compatible with scipy solver)
    x_dot = np.concatenate([
        delt_true_currency,
        delt_eff_currency,
        delt_currency_drift,
        ])
    return x_dot


# this is the impulse response of state to shocks
def evalg(x, t, p, u):
    n = x.shape[0] // 3
    return np.diag(np.concatenate([
        p['sigma'],
        np.zeros([n]),
        np.zeros([n]),
        ]))
