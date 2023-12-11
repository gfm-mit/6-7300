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
    delta_t = 1e-2
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


def plot_golden(x0, p, u, regenerate=False):
    delta_t = 1e-6
    kwargs = dict(
        x0=x0,
        p=p,
        u=u,
        t1=T1,
        delta_t=delta_t,
        f_step=explicit.forward_euler,
    )
    if regenerate:
        xs = list(tqdm(explicit.simulate(**kwargs), total=int(T1 / delta_t)))
        np.save('xs.npy', xs)
    xs = np.load('xs.npy')
    simulation_vis.visualize(xs[::1000], p, u)


def plot_trapezoid_coarsening(x0, p, u):
    golden = np.load('xs.npy')
    results = []
    for delta_t in np.geomspace(1e-4, 1e-1, 4):
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


def plot_dynamic_time_step(x0, p, u):
    golden = np.load('xs.npy')
    results = []
    for delta_t in np.geomspace(1e-5, 1e-1, 5):
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


def plot_rk4_coarsening(x0, p, u):
    golden = np.load('xs.npy')
    results = []
    for delta_t in np.geomspace(1e-5, 1e-1, 5):
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
        plt.plot(subset.delta_t, subset.error, label=g)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title("Trajectory Error")
    plt.xlabel('discretization Δt')
    plt.ylabel('max(Δx)')
    plt.sca(axs[1])
    for g, subset in results.groupby("integrator"):
        plt.plot(subset.delta_t, subset.time, label=g)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title("Integrator Speeds")
    plt.xlabel('discretization Δt')
    plt.ylabel('execution time (seconds)')
    plt.savefig(f'integrators{n}.png')
    plt.show()


if __name__ == '__main__':
    x0, p, u = generate_stochastic_inputs(N)
    if REGENERATE: # comment this bit out once you find a good setting
        np.save('x0.npy', x0)
        np.save('p.npy', p)
    x0 = np.load('x0.npy')
    p = np.load('p.npy', allow_pickle=True)[()]
    test_divergence(x0, p, u)
    if REGENERATE:
        plot_golden(x0, p, u, regenerate=REGENERATE)
    results = []
    results += plot_trapezoid_coarsening(x0, p, u)
    results += plot_dynamic_time_step(x0, p, u)
    results += plot_rk4_coarsening(x0, p, u)
    results = pd.DataFrame(results)
    #res2 = pd.read_csv("sim_results.csv").query("integrator != 'rk4'")
    #results = pd.concat([results, res2])
    results.to_csv("sim_results.csv")
    plot_results(results, N)