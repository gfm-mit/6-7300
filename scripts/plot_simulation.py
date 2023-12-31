import sys
import os
import pathlib

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))


from dynamic import explicit, implicit
from domain_specific.evalf import evalf
from domain_specific.x0 import generate_stochastic_inputs
from domain_specific.jacobian import evalJacobian
from utils import simulation_vis


# yeah, yeah, should probably be 10
T1 = 3
N = 10
REGENERATE = False


def test_divergence(x0, p, u):
    delta_t = 1e-1
    kwargs = dict(
        x0=x0,
        p=p,
        u=u,
        t1=T1,
        delta_t=delta_t,
        f_step=explicit.forward_euler,
    )
    xs = list(tqdm(explicit.simulate(**kwargs), total=int(T1 / delta_t)))
    #xs = np.stack(xs)
    #simulation_vis.visualize(xs, p, u, t=T1)


def plot_golden(x0, p, u, delta_t, regenerate=False):
    if regenerate:
        #delta_t = 1e-4
        kwargs = dict(
            x0=x0,
            p=p,
            u=u,
            t1=T1,
            delta_t=delta_t,
            f_step=explicit.rk4,
        )
        xs = list(tqdm(explicit.simulate(**kwargs), total=int(T1 / delta_t)))
        np.save('xs.npy', xs)
    else:
        xs = np.load('xs.npy')
        simulation_vis.visualize(xs[::1000], p, u)


def plot_trapezoid_coarsening(x0, p, u, delta_t):
    golden = np.load('xs.npy')
    results = []
    #for delta_t in np.geomspace(1e-4, 1e-1, 4):
    if True:
        kwargs = dict(
            x0=x0,
            p=p,
            u=u,
            t1=T1,
            delta_t=delta_t,
            evalf_converter=implicit.get_trapezoid_f,
        )
        tic = time.time()
        xs = list(tqdm(implicit.simulate(**kwargs), total=int(T1 / delta_t)))
        toc = time.time() - tic
        subsample = np.linspace(0, golden.shape[0]-1, len(xs)).astype(int)
        error = np.linalg.norm(golden[subsample] - xs, np.inf, axis=1)
        results += [dict(
            integrator="trap",
            delta_t=delta_t,
            error=np.max(error),
            time=toc)]
    return results


def plot_dynamic_time_step(x0, p, u, delta_t):
    golden = np.load('xs.npy')
    results = []
    #for delta_t in np.geomspace(1e-5, 1e-1, 5):
    if True:
        kwargs = dict(
            x3=x0,
            p=p,
            u=u,
            t1=T1,
            delta_t=delta_t,
            factory=implicit.get_trapezoid_f,
            guess=explicit.rk4,
            dx_error_max=1e-5,
        )
        tic = time.time()
        xs = list(tqdm(implicit.dynamic_step(**kwargs), total=int(T1 / delta_t)))[:]
        toc = time.time() - tic
        golden = np.load('xs.npy')
        subsample = np.linspace(0, golden.shape[0]-1, len(xs)).astype(int)
        error = np.linalg.norm(golden[subsample] - xs, np.inf, axis=1)
        results += [dict(
            integrator="dynamic step",
            delta_t=delta_t,
            error=np.max(error),
            time=toc)]
    return results


def plot_rk4_coarsening(x0, p, u, delta_t):
    golden = np.load('xs.npy')
    results = []
    #for delta_t in np.geomspace(1e-5, 1e-1, 5):
    if True:
        kwargs = dict(
            x0=x0,
            p=p,
            u=u,
            t1=T1,
            delta_t=delta_t,
            f_step=explicit.rk4,
        )
        tic = time.time()
        xs = list(tqdm(explicit.simulate(**kwargs), total=int(T1 / delta_t)))
        toc = time.time() - tic
        subsample = np.linspace(0, golden.shape[0]-1, len(xs)).astype(int)
        error = np.linalg.norm(golden[subsample] - xs, np.inf, axis=1)
        results += [dict(
            integrator="rk4",
            delta_t=delta_t,
            error=np.max(error),
            time=toc)]
    return results


def plot_results(results, n):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    plt.sca(axs[0])
    for g, subset in results.groupby("integrator"):
        plt.plot(subset.n, subset.error, label=g)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title("Trajectory Error")
    plt.xlabel('Number of Countries')
    plt.ylabel('max(Δx)')
    plt.sca(axs[1])
    for g, subset in results.groupby("integrator"):
        plt.plot(subset.n, subset.time, label=g)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title("Integrator Speeds")
    plt.xlabel('Number of Countries')
    plt.ylabel('execution time (seconds)')
    plt.savefig(f'integrators{n}.png')
    plt.show()


def test_n(n):
    x0, p, u = generate_stochastic_inputs(n)
    test_divergence(x0, p, u)
    dt = 1e-4 if n < 50 else 3e-5
    plot_golden(x0, p, u, dt, regenerate=True)
    results = []
    dt = 1e-4 if n < 50 else 3e-5
    results += plot_dynamic_time_step(x0, p, u, dt)
    dt = 1e-3 if n < 50 else 3e-4
    results += plot_trapezoid_coarsening(x0, p, u, dt)
    dt = 1e-2 if n < 50 else 3e-3
    results += plot_rk4_coarsening(x0, p, u, dt)
    for result in results:
        result["n"] = n
    return results

if __name__ == '__main__':
    results = []
    for n in (100, 50, 20, 10, 5, 2):
        results += test_n(n)
    results = pd.DataFrame(results)
    results.to_csv("sim_results.csv")
    plot_results(results, N)
    print(results)