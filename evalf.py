import numpy as np


def generate_inputs(n):
    """
    :param n (int): number of nodes
    :return: state vector x, parameters p, inputs u
    """
    # Placeholders
    y = np.ones([n])                    # n x 1
    tilde_y = np.ones([n])              # n x 1
    mu = np.ones([n])                   # n x 1
    tau1 = 1 * np.ones([n])                 # n x 1
    tau2 = 1 * np.ones([n])                 # n x 1
    sigma = 0.05 * np.ones([n])                # n x 1
    alpha = -1*np.ones([n])                # n x 1
    gamma = np.ones([n])                # n x 1
    d = np.ones([n, n])                 # nm x 1
    g = np.ones([n])                    # n x 1 (little y)
    delt_w = np.ones([n])              # n x 1
    # Build x, p, u arrays
    x = np.array([y, tilde_y, mu])
    p = {'tau1': tau1, 'tau2': tau2, 'sigma': sigma, 'alpha': alpha, 'gamma': gamma, 'd': d, 'g': g}
    u = np.array(delt_w)
    return x, p, u


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
    # Reshape x (had to flatten to make it work with scipy solver)
    n = x.shape[0] // 3
    y, y_tilde, mu = x.reshape(3, n)

    delt_true_currency = np.zeros([n])
    delt_eff_currency = np.zeros([n])
    N = np.zeros([n])
    delt_currency_drift = np.zeros([n])
    g_w = np.sum(p['g'])
    x_ij = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            if i != j:
                x_ij[i, j] = p['g'][i] * p['g'][j] / g_w / p['d'][i, j]
                x_ij[i, j] *= np.power(y_tilde[i] / y_tilde[j], p['gamma'][i])
    for i in range(n):
        delt_true_currency[i] = mu[i] * y[i] + p['sigma'][i] * y[i] * u[i]
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


if __name__ == '__main__':
    n = 2
    t = np.linspace(0, 10, 10)
    x, p, u = generate_inputs(n)
    evalf(x, t, p, u)
