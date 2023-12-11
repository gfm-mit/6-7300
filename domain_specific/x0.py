import numpy as np
import pandas as pd
# only for testing
import sys
import os
import pathlib

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))


def generate_demo_parameters(n, delta_t=1e-2, t=100, seed=5):
    """
    Get time series to simulate shocks
    :param n (int): number of nodes
    :return: state vector x, parameters p, inputs u
    """
    if n > 10:
        print('Can not run for more than 10 real values, so going down to 10 countries only')
        n = 10

    distances = pd.read_parquet('domain_specific/distances.parquet')
    tarrifs = pd.read_parquet('domain_specific/tarrifs.parquet')
    sigmas = pd.read_parquet('domain_specific/volatilities.parquet')
    # Not sure why we're using parquet for this
    # But doing it anyway for consistency
    gdps = pd.read_parquet('domain_specific/gdp.parquet')

    tau1 = 0.25
    tau2 = 0.25
    tau3 = 1
    sigma = (0.1 * sigmas.iloc[:n]).to_numpy() # Doesn't do anything without Weiner process
    alpha = 2e-1
    gamma2 = 1 * np.ones([n])

    np.random.seed(seed)
    d_timeseries = np.ones([int(t / delta_t), n, n])
    # Use tariffs and distance as a proxy for trade friction
    d = np.zeros([n, n])
    for c in range(n):
        for c_other in range(n):
            d[c, c_other] = np.exp(tarrifs['tarrif'].iloc[c] / np.max(tarrifs['tarrif'])) * (distances.iloc[c, c_other] / 24902)

    # Time series to simulate shock in demo
    for i in range(int(t / delta_t)):
        d_timeseries[i, :, :] = d
        # Shock at t=5000
        if i > 5000:
            d_timeseries[i, 2, 0] = 5e-2
            d_timeseries[i, 0, 2] = 5e-2

    g = (gdps['gdp'].to_numpy()[:n] / gdps['gdp'].to_numpy()[:n].max())

    # Build x, p, u arrays
    p = {'tau1': tau1, 'tau2': tau2, 'tau3': tau3, 'sigma': sigma, 'alpha': alpha, 'gamma2': gamma2, 'd': d_timeseries, 'g': g}
    return p


def generate_real_parameters(n):
    """
    :param n (int): number of nodes
    :return: state vector x, parameters p, inputs u
    """
    if n > 10:
        print('Can not run for more than 10 real values, so going down to 10 countries only')
        n = 10

    distances = pd.read_parquet('domain_specific/distances.parquet')
    sigmas = pd.read_parquet('domain_specific/volatilities.parquet')
    # Not sure why we're using parquet for this
    # But doing it anyway for consistency
    gdps = pd.read_parquet('domain_specific/gdp.parquet')

    tau1 = 1
    tau2 = 1
    tau3 = 1
    sigma = (0.1 * sigmas.iloc[:n]).to_numpy() # doesn't do anything without Weiner process
    alpha = 1e-1
    gamma2 = 1 * np.ones([n]) # values between 0 and 1 don't seem to affect divergence

    #d = distances.iloc[:n, :n].to_numpy()
    d = np.ones([n, n])
    d[1, 0] = 0.1
    d[0, 1] = 0.1
    #d = (1.5 + d/d.max().max()).to_numpy()-1.5 * np.eye(n) # Get realistic values of distances in range of 0.5 to 1.5
    #d = d / d.max()
    print(d)

    g = (gdps['gdp'].to_numpy()[:n] / gdps['gdp'].to_numpy()[:n].max())

    #g = np.random.lognormal(size=n, sigma=2.5)
    #d = np.exp(np.random.uniform(-1, 1, size=[n, n]))

    # Build x, p, u arrays
    p = {'tau1': tau1, 'tau2': tau2, 'tau3': tau3, 'sigma': sigma, 'alpha': alpha, 'gamma2': gamma2, 'd': d, 'g': g}
    return p


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
    tau1 = 1
    tau2 = 1
    tau3 = 1
    sigma = 1e-2 * np.ones([n])
    alpha = 1e-1
    gamma2 = 1 * np.ones([n]) # values between 0 and 1 don't seem to affect divergence
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
    p['g'] = np.random.lognormal(size=n, sigma=2.5)
    p['d'] = np.exp(np.random.uniform(-1, 1, size=[n, n]))

    y = np.random.normal(size=n, scale=0.5)
    y_tilde = y + np.random.normal(size=n, scale=0.01)
    mu = np.random.normal(size=n, scale=0.01)
    x0 = np.stack([y, y_tilde, mu])

    return x0, p, generate_shocks(n)


def generate_deterministic_real_inputs(n):
    return generate_default_state(n), generate_real_parameters(n), generate_shocks(n)


def generate_stochastic_real_inputs(n):
    p = generate_real_parameters(n)

    y = np.random.normal(size=n, scale=0.5)
    y_tilde = y + np.random.normal(size=n, scale=0.01)
    mu = np.random.normal(size=n, scale=0.01)
    x0 = np.stack([y, y_tilde, mu])

    return x0, p, generate_shocks(n)


def generate_demo_inputs(n, t=100, seed=5):
    p = generate_demo_parameters(n, t=t)

    np.random.seed(seed)
    y = np.random.normal(size=n, scale=0.5)
    y_tilde = y + np.random.normal(size=n, scale=0.05)
    mu = np.random.normal(size=n, scale=0.05)
    x0 = np.stack([y, y_tilde, mu])

    return x0, p, generate_shocks(n)
