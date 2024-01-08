import sys
import os
import pathlib

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))


import newton.from_julia
import domain_specific.demo
from dynamic import explicit, implicit
from domain_specific.evalf import evalf
from domain_specific.x0 import generate_stochastic_inputs
from domain_specific.jacobian import evalJacobian
from utils import simulation_vis


def get_delta_ts():
    return np.geomspace(1e-4, 1e-1, 4)


def plot_golden(x0, p, u, regenerate=False):
    delta_t = 1e-5
    t1 = 40
    kwargs = dict(
        x0=x0,
        p=p,
        u=u,
        t1=t1,
        delta_t=delta_t,
        f_step=explicit.rk4,
    )
    if regenerate:
        xs = list(tqdm(explicit.simulate(**kwargs), total=int(t1 / delta_t)))
        np.save('xs.npy', xs)
    xs = np.load('xs.npy')
    simulation_vis.visualize(xs[::1000], p, u, savefig='c.1.png')
    assert False


def plot_trapezoid_coarsening(x0, p, u):
    golden = np.load('xs.npy')
    for delta_t in get_delta_ts():
        t1 = 40
        subsample = int(np.round(1e5 * delta_t))
        subsample = np.s_[::subsample]
        kwargs = dict(
            x0=x0,
            p=p,
            u=u,
            t1=t1,
            delta_t=delta_t,
            evalf_converter=implicit.get_trapezoid_f,
        )
        tic = time.time()
        xs = list(tqdm(implicit.simulate(**kwargs), total=int(t1 / delta_t), desc=f"trap {delta_t}"))
        toc = time.time() - tic
        subsample = np.linspace(0, golden.shape[0]-1, len(xs)).astype(int)
        error = np.linalg.norm(golden[subsample] - xs, np.inf, axis=1)
        error = np.max(error)
        yield "trap", delta_t, toc, error


def plot_dynamic_coarsening(x0, p, u, k):
    golden = np.load('xs.npy')
    for delta_t in get_delta_ts():
        t1 = 40
        dx_error_max = np.power(10., -k)
        kwargs = dict(
            x3=x0,
            p=p,
            u=u,
            t1=t1,
            delta_t=delta_t,
            factory=implicit.get_trapezoid_f,
            dx_error_max=dx_error_max,
        )
        tic = time.time()
        xs = list(tqdm(implicit.dynamic_step(**kwargs), total=int(t1 / delta_t), desc=f"dynamic {k} {dx_error_max}"))
        toc = time.time() - tic
        subsample = np.linspace(0, golden.shape[0]-1, len(xs)).astype(int)
        error = np.linalg.norm(golden[subsample] - xs, np.inf, axis=1)
        error = np.max(error)
        yield f"dynamic {k}", delta_t, toc, error

def plot_fe_coarsening(x0, p, u):
    golden = np.load('xs.npy')
    for delta_t in get_delta_ts():
        t1 = 40
        kwargs = dict(
            x0=x0,
            p=p,
            u=u,
            t1=t1,
            delta_t=delta_t,
        )
        tic = time.time()
        xs = list(tqdm(explicit.simulate(**kwargs), total=int(t1 / delta_t), desc=f"fe {delta_t}"))
        toc = time.time() - tic
        subsample = np.linspace(0, golden.shape[0]-1, len(xs)).astype(int)
        error = np.linalg.norm(golden[subsample] - xs, np.inf, axis=1)
        error = np.max(error)
        yield "fe", delta_t, toc, error


def plot_rk4_coarsening(x0, p, u):
    golden = np.load('xs.npy')
    for delta_t in get_delta_ts():
        t1 = 40
        kwargs = dict(
            x0=x0,
            p=p,
            u=u,
            t1=t1,
            delta_t=delta_t,
            f_step=explicit.rk4,
        )
        tic = time.time()
        xs = list(tqdm(explicit.simulate(**kwargs), total=int(t1 / delta_t), desc=f"rk4 {delta_t}"))
        toc = time.time() - tic
        subsample = np.linspace(0, golden.shape[0]-1, len(xs)).astype(int)
        error = np.linalg.norm(golden[subsample] - xs, np.inf, axis=1)
        error = np.max(error)
        yield "rk4", delta_t, toc, error


def jank(x0, p, u):
    p_initial = p.copy()
    p_initial['d'] = p_initial['d'][0, :, :]
    x1 = newton.from_julia.newton_julia_jacobian_free_wrapper(x0, p_initial, u)
    delta_t = 1e-2
    t1 = 30
    kwargs = dict(
        x0=x1,
        p=p,
        u=u,
        t1=t1,
        delta_t=delta_t,
        f_step=explicit.rk4,
    )
    tic = time.time()
    xs = list(tqdm(explicit.simulate(**kwargs), total=int(t1 / delta_t)))
    return time.time() - tic


if __name__ == '__main__':
    x0, p, u = domain_specific.demo.generate_wobble_inputs(3)
    #if True: # comment this bit out once you find a good setting
    #    np.save('x0.npy', x0)
    #    np.save('p.npy', p)
    #x0 = np.load('x0.npy')
    #p = np.load('p.npy', allow_pickle=True)[()]
    print(jank(x0, p, u))
    ##plot_golden(x0, p, u, regenerate=True)
    ##results = []
    ##results += list(plot_dynamic_coarsening(x0, p, u, 4))
    ##results += list(plot_dynamic_coarsening(x0, p, u, 5))
    ##results += list(plot_fe_coarsening(x0, p, u))
    ##results += list(plot_trapezoid_coarsening(x0, p, u))
    ##results += list(plot_rk4_coarsening(x0, p, u))
    ##results = pd.DataFrame(results)
    ##results.to_csv('error_speed.csv')
    #results = pd.read_csv('error_speed.csv')
    #results.columns = "idx integrator delta_t time error".split()
    #results.delta_t = results.delta_t.astype(float)
    #results = results.query('integrator != "dynamic 4"')
    #results.integrator = results.integrator.str.replace("dynamic 5", "Trapezoid - Dynamic")
    #results.integrator = results.integrator.str.replace("fe", "Forward Euler")
    #results.integrator = results.integrator.str.replace("rk4", "Runge-Kutta 4")
    #results.integrator = results.integrator.str.replace("trap", "Trapezoid - Fixed")
    #for idx, df in results.groupby('integrator'):
    #    color = plt.plot(df.time, df.error, label=idx, zorder=-10)[0].get_color()
    #    plt.scatter(df.time, df.error, color="white", s=700, zorder=0)
    #    for idx2, row in df.iterrows():
    #        exponent = str(int(np.round(np.log10(row.delta_t))))
    #        text = f"$10^{{{exponent}}}$"
    #        if row.delta_t == .1 and idx == "Runge-Kutta 4":
    #            text = f"$\Delta t = 10^{{{exponent}}}$"
    #        plt.text(row.time, row.error, text, ha='center', va='center', color=color)
    #plt.legend()
    #plt.xscale('log')
    #plt.xlabel('Execution Time (seconds)')
    #plt.yscale('log')
    #plt.ylabel('Error')
    #plt.title('Error vs. Execution Time for Various Integrators')
    #plt.tight_layout()
    #plt.show()
    #print(results)