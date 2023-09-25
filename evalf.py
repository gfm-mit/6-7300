import numpy as np


def generate_inputs(n, E):
    """
    :param n (int): number of nodes
    :return: state vector x, parameters p, inputs u
    """
    # Placeholders
    y = np.zeros((n, 1))                    # n x 1
    tilde_y = np.zeros((n, 1))              # n x 1
    mu = np.zeros((n, 1))                   # n x 1
    tau1 = np.zeros((n, 1))                 # n x 1
    tau2 = np.zeros((n, 1))                 # n x 1
    omega = np.zeros((n, 1))                # n x 1
    alpha = np.zeros((n, 1))                # n x 1
    gamma2 = np.zeros((E.shape[0], 1))      # nm x 1
    d = np.zeros((E.shap[0], 1))            # nm x 1
    g = np.zeros((n, 1))                    # n x 1 (little y)
    delt_w = np.zeros((n, 1))               # n x 1
    # Build x, p, u arrays
    x = np.array([y, tilde_y, mu])
    p = np.array([tau1, tau2, omega, alpha, gamma2, d, g])
    u = np.array([delt_w])
    return x, p, u


def evalf(x, p, u, E):
    """
    Removed gamma1 and nu
    Removed y_w, P_i and P_j

    :param x: state vector
    [y, tilde_y, mu]
    - y = true currency
    - tilde_y = effective currency
    - mu = currency drift
    :param p: parameters
    [tau1, tau2, omega, alpha, gamma2, d, v, g]
    - tau1 = delay between true and effective currency
    - tau2 = delay between trade imbalance and currency drift
    - omega = strength of market volatility
    - alpha = converts trade imbalance into monetary value
    - gamma2 = tuning parameter of BTR
    - d = distance
    - v = value differential parameter
    - g = GDP at each node
    :param u: inputs
    [delt_w]
    :return: delt_x = f(x, p, u)
    """
    delt_true_currency = get_delt_true_currency(mu=x[2], omega=p[2], delt_W=u[0], Y=x[0])
    delt_eff_currency = get_eff_currency(Y=x[0], tilde_Y=x[1], tau1=p[0])
    delt_currency_drift = get_currency_drift(tilde_Y=x[1], alpha=p[3], tau2=p[1], E=E)
    return np.array([delt_true_currency, delt_eff_currency, delt_currency_drift])


def get_delt_true_currency(mu, omega, delt_W, Y):
    """
    delta_Y = (mu + alpha .* delt_W) .* Y
    """
    return mu + np.multiply((np.multiply(omega, delt_W)), Y)


def get_eff_currency(Y, tilde_Y, tau1):
    """
    delta_Y_tilde = (Y - tilde_Y) ./ tau1
    """
    return np.divide((Y - tilde_Y), tau1)


def get_currency_drift(tilde_Y, alpha, tau2, E):
    """
    delt_mu = (alpha .* N - mu) ./ tau2
    """
    lambda_sum = get_lambda_sum(E)
    gamma_select = get_gamma_select(E)
    x = get_X(lambda_sum, gamma_select, y)
    q = get_q(A, x)
    N = get_N(x, tilde_Y, lambda_sum, gamma_select, q)
    return np.divide(np.multiply(alpha, N), tau2)


def get_lambda_sum(E):
    return 0.5 * (np.absolute(E) - E)


def get_gamma_select(E):
    return 0.5 * (E + np.absolute(E))


def get_N(x, tilde_Y, lambda_sum, gamma_select, q):
    export_flow = lambda_sum @ x
    import_flow = np.divide(lambda_sum @ np.multiply((gamma_select @ tilde_Y), q), tilde_Y)
    return export_flow - import_flow


def get_X(lambda_sum, gamma_select, y):
    R = get_R()
    return np.divide(, R)


def get_q(A, x):
    return A @ x


def get_R():
    return


def get_E(config):
    E = []
    with open(config, 'r') as f:
        for line in f.readlines():
            E.append(line.strip('\n').split('\t'))
    return np.array(E)


if __name__ == '__main__':
    n = 3
    E = get_E('test.txt')
    x, p, u = generate_inputs(n, E)
    evalf(x, p, u)
