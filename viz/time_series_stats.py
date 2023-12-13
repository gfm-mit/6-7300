import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

import sys
import os
import pathlib

from tqdm import tqdm

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
from domain_specific.x0 import generate_demo_inputs
from dynamic import explicit
import newton.from_julia


def get_param_final_delta(x0, p, u, param, k):
    p_initial = p.copy()
    p_initial['d'] = p['d'][0, :, :]
    p_initial[param] = p[param] * k
    x1 = newton.from_julia.newton_julia_jacobian_free_wrapper(x0, p_initial, u)

    p_final = p.copy()
    p_final['d'] = p['d'][-1, :, :]
    p_final[param] = p[param] * k
    x2 = newton.from_julia.newton_julia_jacobian_free_wrapper(x0, p_final, u)

    delta = x2 - x1
    delta = delta[2] - delta[0]
    return delta


def tune_final_params(x0, p, u):
    for param in "tau1 tau2 tau3 alpha".split():
        results = []
        for k in np.geomspace(7e-1, 2e0, 20):
            delta = get_param_final_delta(x0, p, u, param, k)
            results += [dict(
                k=k,
                delta=delta,
                noise_x=1e-4*np.random.normal(),
                noise_y=1e-4*np.random.normal(),
            )]
        results = pd.DataFrame(results)
        plt.plot(results.k, results.delta, label=param)
        plt.scatter(results.k + results.noise_x, results.delta + results.noise_y)

    delta = get_param_final_delta(x0, p, u, "alpha", 1)
    plt.legend()
    plt.xscale('log')
    plt.title("final:{:.2f}".format(delta))


def tune_wobble_params(x0, p, u):
    for param in "tau1 tau2 tau3 alpha".split():
        results = []
        for k in tqdm(np.geomspace(1e-2, 1e2, 20)):
            try:
                delta = get_param_wobble_delta(x0, p, u, param, k)
            except AssertionError:
                delta = np.inf
            delta = np.clip(delta, 0, 1e1)

            results += [dict(
                k=k,
                delta=delta,
                noise_x=1e-2*np.random.normal(),
                noise_y=1e-2*np.random.normal(),
            )]
        results = pd.DataFrame(results)
        plt.plot(results.k, results.delta, label=param)
        plt.scatter(results.k + results.noise_x, results.delta + results.noise_y)
    delta = get_param_wobble_delta(x0, p, u, "alpha", 1)
    plt.legend()
    plt.xscale('log')
    plt.title("wobble:{:.2f}".format(delta))


def get_param_wobble_delta(x0, p, u, param, k):
    p_initial = p.copy()
    p_initial['d'] = p_initial['d'][0, :, :]
    p_initial[param] *= k
    x1 = newton.from_julia.newton_julia_jacobian_free_wrapper(x0, p_initial, u)

    p_final = p.copy()
    p_final[param] *= k

    t1 = 200
    kwargs = dict(
                x0=x1,
                p=p_final,
                u=u,
                t1=t1,
                delta_t=2e-1,
                f_step=explicit.rk4,
                demo=True
            )
    xs = np.array(list(explicit.simulate(**kwargs)))
    xs = np.stack(xs)
    usd = xs[:, 0]
    inr = xs[:, 2]
    delta = inr - usd
    delta = delta - delta[0]
    delta = np.abs(delta)
    return argrelextrema(delta, np.greater)[0].shape[0]


def tune_slope_params(x0, p, u):
    for param in "tau1 tau2 tau3 alpha".split():
        if param == "alpha":
            continue
        results = []
        for k in tqdm(np.geomspace(1e-2, 1e2, 20)):
            delta = get_param_slope_delta(x0, p, u, param, k)

            results += [dict(
                k=k,
                delta=delta,
                noise_x=1e-3*np.random.normal(),
                noise_y=1e-3*np.random.normal(),
            )]
        results = pd.DataFrame(results)
        plt.plot(results.k, results.delta, label=param)
        plt.scatter(results.k + results.noise_x, results.delta + results.noise_y)
    delta = get_param_slope_delta(x0, p, u, "alpha", 1)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.title("slope:{:.2f}".format(delta))


def get_param_slope_delta(x0, p, u, param, k):
    p_initial = p.copy()
    p_initial['d'] = p_initial['d'][0, :, :]
    p_initial[param] *= k
    x1 = newton.from_julia.newton_julia_jacobian_free_wrapper(x0, p_initial, u)

    p_final = p.copy()
    p_final[param] *= k

    t1 = 60
    delta_t = 6e-2
    kwargs = dict(
                x0=x1,
                p=p_final,
                u=u,
                t1=t1,
                delta_t=delta_t,
                f_step=explicit.rk4,
                demo=True
            )
    xs = np.array(list(explicit.simulate(**kwargs)))
    xs = np.stack(xs)
    usd = xs[:, 0]
    inr = xs[:, 2]
    delta = inr - usd
    delta = delta - delta[0]
    delta = np.abs(delta[-1] - delta[-2]) / delta_t
    return delta


def multi_tune(x0, p, u):
    fig, axs = plt.subplots(3)
    plt.sca(axs[0])
    tune_slope_params(x0, p, u)
    plt.sca(axs[1])
    tune_final_params(x0, p, u)
    plt.sca(axs[2])
    tune_wobble_params(x0, p, u)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Generate data to visualize
    x0, p, u = generate_demo_inputs(10)
    #np.random.seed()
    multi_tune(x0, p, u)