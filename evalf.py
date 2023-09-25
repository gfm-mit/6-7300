import numpy as np


def generate_inputs(n):
    """
    :param n (int): number of nodes
    :return: state vector x, parameters p, inputs u
    """
    x = np.zeros()
    p = np.zeros()
    u = np.zeros()
    return x, p, u


def evalf(x, p, u):
    """
    :param x: state vector
    [y, tilde_y, mu]
    :param p: parameters
    [tau1, tau2, omega, alpha, phi1, phi2, d, v, y_i, y_w, P_i]
    :param u: inputs
    [delt_w, delt_d, delt_v]
    :return: delt_x = f(x, p, u)
    """
    delt_true_currency = get_delt_true_currency()
    delt_eff_currency = get_eff_currency()
    delt_currency_drift = get_currency_drift()
    return np.array([delt_true_currency, delt_eff_currency, delt_currency_drift])


def get_delt_true_currency():

    return np.multiply((mu + np.multiplu(omega, delt_W)), Y)


def get_eff_currency():
    return np.divide((Y - tilde_Y), tau1)


def get_currency_drift():
    N = get_N()
    return np.divide(np.multiply(alpha, N) - mu, tau2)


def get_N():
    x = get_X()
    return


def get_X():
    R = get_R()
    return


def get_R():
    return


if __name__ == '__main__':
    n = 3
    x, p, u = generate_inputs(n)
    evalf(x, p, u)
