import numpy as np


def generate_inputs(n):
    """
    :param n (int): number of nodes
    :return: state vector x, parameters p, inputs u
    """
    # Placeholders
    # y = 10000 * np.random.random([n])                    # n x 1
    # tilde_y = y                    # n x 1
    # tilde_y = 10000 * np.random.random([n])                    # n x 1
    y = np.ones([n])                    # n x 1
    tilde_y = np.ones([n])              # n x 1
    mu = np.ones([n])                   # n x 1
    tau1 = 1 * np.ones([n])             # n x 1
    tau2 = 1 * np.ones([n])             # n x 1
    sigma = 1e-3 * np.ones([n])         # n x 1
    alpha = -1*np.ones([n])             # n x 1
    gamma2 = 1 * np.ones([n])          # n x 1
    d = np.ones([n, n])                 # nm x 1
    # d = 1000 * np.random.random([n, n])                 # nm x 1
    g = np.ones([n])                    # n x 1 (little y)
    gw = np.sum(g)
    delt_w = np.zeros([n])              # n x 1
    # Build x, p, u arrays
    x = np.array([y, tilde_y, mu])
    p = {'tau1': tau1, 'tau2': tau2, 'sigma': sigma, 'alpha': alpha, 'gamma2': gamma2, 'd': d, 'g': g, 'gw':gw}
    u = np.array(delt_w)
    return x, p, u

def generate_random_parameters(n):
    """
    :param n (int): number of nodes
    :return: state vector x, parameters p, inputs u
    """
    x, p, u = generate_inputs(n)
    p['d'] = np.exp(np.random.uniform(-1, 1, size=[n, n]))
    return x, p, u

def generate_parameter_ranges(n, samples):
    for _ in range(samples):
        yield generate_random_parameters(n)

def generate_lognormal_input(n):
    x0, p, u = generate_inputs(n)
    x0 = np.random.lognormal(size=[3, n])
    return x0, p, u