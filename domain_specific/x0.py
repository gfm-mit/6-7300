import numpy as np


def generate_default_state(n):
    y = np.zeros([n])                    # n x 1
    tilde_y = np.zeros([n])              # n x 1
    mu = np.zeros([n])                   # n x 1
    return np.array([y, tilde_y, mu])

def generate_parameters(n):
    """
    :param n (int): number of nodes
    :return: state vector x, parameters p, inputs u
    """
    # Placeholders
    # y = 10000 * np.random.random([n])                    # n x 1
    # tilde_y = y                    # n x 1
    # tilde_y = 10000 * np.random.random([n])                    # n x 1
    tau1 = 1
    tau2 = 1
    tau3 = 1e3
    sigma = 1e-2 * np.ones([n])
    alpha = 5e-1
    gamma2 = 1 * np.ones([n])          # n x 1
    d = np.ones([n, n])                 # nm x 1
    # d = 1000 * np.random.random([n, n])                 # nm x 1
    g = np.ones([n])                    # n x 1 (little y)
    # Build x, p, u arrays
    p = {'tau1': tau1, 'tau2': tau2, 'tau3': tau3, 'sigma': sigma, 'alpha': alpha, 'gamma2': gamma2, 'd': d, 'g': g}
    return p

def generate_shocks(n):
    return np.zeros([n])              # n x 1

def generate_deterministic_inputs(n):
    return generate_default_state(n), generate_parameters(n), generate_shocks(n)

def generate_stochastic_inputs(n):
    p = generate_parameters(n)
    #p['g'] = np.random.lognormal(size=n, sigma=2.5)
    #p['d'] = np.exp(np.random.uniform(-1, 1, size=[n, n]))

    y = np.random.normal(size=n, scale=0.01)
    y_tilde = y + np.random.normal(size=n, scale=0.01)
    mu = np.random.normal(size=n, scale=0.01)
    x0 = np.stack([y, y_tilde, mu])

    return x0, p, generate_shocks(n)