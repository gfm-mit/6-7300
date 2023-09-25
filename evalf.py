import numpy as np


def generate_inputs(n, E):
    """
    :param n (int): number of nodes
    :return: state vector x, parameters p, inputs u
    """
    # Placeholders
    y = np.ones((n, 1))                    # n x 1
    tilde_y = np.ones((n, 1))              # n x 1
    mu = np.ones((n, 1))                   # n x 1
    tau1 = np.ones((n, 1))                 # n x 1
    tau2 = np.ones((n, 1))                 # n x 1
    omega = np.ones((n, 1))                # n x 1
    alpha = np.ones((n, 1))                # n x 1
    d = np.ones((E.shape[0], 1))           # nm x 1
    g = np.ones((n, 1))                    # n x 1 (little y)
    delt_w = np.ones((n, 1))               # n x 1
    # Build x, p, u arrays
    x = np.array([y, tilde_y, mu])
    p = {'tau1': tau1, 'tau2': tau2, 'omega': omega, 'alpha': alpha, 'd': d, 'g': g}
    u = np.array([delt_w])
    return x, p, u


def evalf(x, t, p, u, E):
    """
    Removed gamma1 and nu
    Removed y_w, P_i and P_j

    :param x (array): state vector
    [y, tilde_y, mu]
    - y = true currency
    - tilde_y = effective currency
    - mu = currency drift
    :param p (dict): parameters
    [tau1, tau2, omega, alpha, d, v, g]
    - tau1 = delay between true and effective currency
    - tau2 = delay between trade imbalance and currency drift
    - omega = strength of market volatility
    - alpha = converts trade imbalance into monetary value
    - d = distance
    - v = value differential parameter
    - g = GDP at each node
    :param u (array): inputs
    [delt_w]
    :return: delt_x = f(x, p, u)
    """
    # Reshape x (had to flatten to make it work with scipy solver)
    x = x.reshape(3, 3, 1)

    # Compute components of x
    delt_true_currency = get_delt_true_currency(mu=x[2], omega=p['omega'], delt_W=u[0], Y=x[0])
    delt_eff_currency = get_delt_eff_currency(Y=x[0], tilde_Y=x[1], tau1=p['tau1'])
    delt_currency_drift = get_delt_currency_drift(tilde_Y=x[1], alpha=p['alpha'], tau2=p['tau2'], g=p['g'], E=E, d=p['d'])

    # Flatten X (to make compatible with scipy solver)
    x_dot = np.array([delt_true_currency, delt_eff_currency, delt_currency_drift]).reshape(9,)
    return x_dot


def get_delt_true_currency(mu, omega, delt_W, Y):
    """
    delta_Y = (mu + alpha .* delt_W) .* Y
    """
    return np.multiply(mu + np.multiply(omega, delt_W), Y)


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
    return 0.5 * (np.absolute(E) - E)


def get_gamma_select(E):
    return 0.5 * (E + np.absolute(E))


def get_N(x, tilde_Y, lambda_sum, gamma_select, q):
    export_flow = np.transpose(lambda_sum) @ x
    import_flow = np.divide(np.transpose(lambda_sum) @ np.multiply((gamma_select @ tilde_Y), q), tilde_Y)
    return export_flow - import_flow


def get_X(lambda_sum, gamma_select, g, d, E, tilde_Y):
    R = get_R(d, E, tilde_Y)
    return np.divide(np.multiply(lambda_sum @ g, gamma_select @ g), R)


def get_q(A, x):
    return A @ x


def get_R(d, E, tilde_Y):
    exponential = np.exp(E @ tilde_Y)
    return np.multiply(d, exponential)


def get_E(config):
    E = []
    with open(config, 'r') as f:
        for line in f.readlines():
            E.append([float(x) for x in line.strip('\n').split('\t')])
    return np.array(E)


if __name__ == '__main__':
    n = 3
    E = get_E('configs/test.txt')
    t = np.linspace(0, 10, 10)
    x, p, u = generate_inputs(n, E)
    evalf(x, t, p, u, E)
