import numpy as np
import einops


def evalf(x, t, p, u):
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
    assert t is None
    assert isinstance(p, dict)
    # Reshape x (had to flatten to make it work with scipy solver)
    n = x.shape[0] // 3
    y, y_tilde, mu = x.reshape(3, n)

    # initialize node quantities
    delt_true_currency = np.zeros([n])
    delt_eff_currency = np.zeros([n])
    N = np.zeros([n])
    delt_currency_drift = np.zeros([n])

    # initialize derived parameter
    g_w = np.sum(p['g'])

    # initialize component quantities
    x_ij = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            if i != j:
                x_ij[i, j] = p['g'][i] * p['g'][j] / g_w / p['d'][i, j]
                x_ij[i, j] *= np.power(y_tilde[i] / y_tilde[j], p['gamma2'][i])
    
    # update node quantities
    for i in range(n):
        delt_true_currency[i] = mu[i] * y[i]
        delt_eff_currency[i] = (y[i] - y_tilde[i]) / p['tau1'][i]
        exports = x_ij[i]
        imports = x_ij[:, i] * y_tilde[i] / y_tilde
        N[i] = np.sum(exports) - np.sum(imports)
        delt_currency_drift[i] = (p['alpha'][i] * N[i] - mu[i]) / p['tau2'][i]

    # Flatten X (to make compatible with scipy solver)
    x_dot = np.concatenate([
        delt_true_currency,
        delt_eff_currency,
        delt_currency_drift,
        ])
    return x_dot


def evalg(x, t, p, u):
    n = x.shape[0] // 3
    y, y_tilde, mu = x.reshape(3, n)
    return np.diag(np.concatenate([
        p['sigma'] * y,
        0 * y_tilde,
        0 * mu
        ]))
