import numpy as np


def generate_inputs(n, E):
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


def evalf(x, t, p, u, E, debug=False):
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
    if debug:
        print(x_ij)
    for i in range(n):
        delt_true_currency[i] = mu[i] * y[i] + p['sigma'][i] * y[i] * u[i]
        delt_eff_currency[i] = (y[i] - y_tilde[i]) / p['tau1'][i]
        exports = x_ij[i]
        imports = x_ij[:, i] * y_tilde[i] / y_tilde
        N[i] = np.sum(exports) - np.sum(imports)
        if debug:
            print((p['alpha'][i] * N[i] - mu[i]) / p['tau2'][i])
        delt_currency_drift[i] = (p['alpha'][i] * N[i] - mu[i]) / p['tau2'][i]
    #print(delt_true_currency, delt_eff_currency, delt_currency_drift)

    # Flatten X (to make compatible with scipy solver)
    x_dot = np.concatenate([
        delt_true_currency,
        delt_eff_currency,
        delt_currency_drift,
        ])
    return x_dot


def get_delt_true_currency(mu, sigma, delt_W, Y):
    """
    delta_Y = (mu + alpha .* delt_W) .* Y
    """
    return np.multiply(mu + np.multiply(sigma, delt_W), Y)


def get_delt_eff_currency(Y, tilde_Y, tau1):
    """
    delta_Y_tilde = (Y - tilde_Y) ./ tau1
    """
    return np.divide((Y - tilde_Y), tau1)


def get_delt_currency_drift(tilde_Y, alpha, tau2, g, E, d):
    """
    delt_mu = (alpha .* N - mu) ./ tau2
    """
    lambda_sum = get_lambda_sum(E)
    gamma_select = get_gamma_select(E)
    A = np.ones((E.shape[0], E.shape[0]))
    x = get_X(lambda_sum, gamma_select, g, d, E, tilde_Y)
    q = get_q(A, x)
    N = get_N(x, tilde_Y, lambda_sum, gamma_select, q)
    return np.divide(np.multiply(alpha, N), tau2)


def get_lambda_sum(E):
    return 0.5 * (E + np.absolute(E))


def get_gamma_select(E):
    return 0.5 * (np.absolute(E) - E)


def get_N(x, tilde_Y, lambda_sum, gamma_select, q):
    export_flow = np.transpose(lambda_sum) @ x
    import_flow = np.divide(np.transpose(lambda_sum) @ np.multiply((gamma_select @ tilde_Y), q), tilde_Y)
    #print(export_flow)
    #print(import_flow)
    return export_flow - import_flow


def get_X(lambda_sum, gamma_select, g, d, E, tilde_Y):
    R = get_R(d, E, tilde_Y, lambda_sum, gamma_select)
    return np.divide(np.multiply(lambda_sum @ g, gamma_select @ g), R)


def get_q(A, x):
    return A @ x


def get_R(d, E, tilde_Y, lambda_sum, gamma_select, gamma=1):
    price = (lambda_sum  @ tilde_Y) / (gamma_select @ tilde_Y)
    price = np.power(price, gamma)
    return np.multiply(d, price)


def get_E(config):
    E = []
    with open(config, 'r') as f:
        for line in f.readlines():
            E.append([float(x) for x in line.strip('\n').split('\t')])
    return np.array(E)


if __name__ == '__main__':
    n = 2
    E = get_E('configs/test.txt')
    t = np.linspace(0, 10, 10)
    x, p, u = generate_inputs(n, E)
    evalf(x, t, p, u, E)
