import sys
import os
import pathlib

import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))


from dynamic import explicit, implicit
from domain_specific.evalf import evalf
from domain_specific.x0 import generate_stochastic_inputs
from domain_specific.jacobian import evalJacobian
from utils import simulation_vis


def plot_golden(x0, p, u, regenerate=False):
    delta_t = 1e-5
    t1 = 40
    kwargs = dict(
        x0=x0,
        p=p,
        u=u,
        t1=t1,
        delta_t=delta_t,
        f_step=explicit.forward_euler,
    )
    if regenerate:
        xs = list(tqdm(explicit.simulate(**kwargs), total=int(t1 / delta_t)))
        np.save('xs.npy', xs)
    xs = np.load('xs.npy')
    simulation_vis.visualize(xs[::1000], p, u, savefig='c.1.png')

def plot_forward_euler_coarsening(x0, p, u):
    golden = np.load('xs.npy')
    for delta_t in np.geomspace(3e-5, 3e-1, 5):
        t1 = 40
        kwargs = dict(
            x0=x0,
            p=p,
            u=u,
            t1=t1,
            delta_t=delta_t,
            f_step=explicit.forward_euler,
        )
        tic = time.time()
        xs = list(tqdm(explicit.simulate(**kwargs), total=int(t1 / delta_t)))
        toc = time.time() - tic
        subsample = np.linspace(0, golden.shape[0]-1, len(xs)).astype(int)
        error = np.linalg.norm(golden[subsample] - xs, np.inf, axis=1)
        ts = np.linspace(0, 40, error.shape[0])
        plt.plot(ts, error, label=f'$\Delta t = {delta_t:.1e}$, {toc:.3f} seconds')
    plt.ylabel('$||x_{\Delta t_i} - x_{10^{-5}}||$')
    plt.yscale('log')
    plt.xlabel('Time')
    plt.legend()
    plt.ylim([1e-5, 1e2])
    plt.savefig('c.2.png')
    plt.show()

def plot_trapezoid_coarsening(x0, p, u):
    golden = np.load('xs.npy')
    for delta_t in np.geomspace(1e-2, 1e1, 7):
        t1 = 40
        subsample = int(np.round(1e5 * delta_t))
        subsample = np.s_[::subsample]
        kwargs = dict(
            x0=x0,
            p=p,
            u=u,
            t1=t1,
            delta_t=delta_t,
            factory=implicit.get_trapezoid_f,
        )
        tic = time.time()
        xs = list(tqdm(implicit.simulate(**kwargs), total=int(t1 / delta_t)))
        toc = time.time() - tic
        subsample = np.linspace(0, golden.shape[0]-1, len(xs)).astype(int)
        error = np.linalg.norm(golden[subsample] - xs, np.inf, axis=1)
        ts = np.linspace(0, 40, error.shape[0])
        plt.plot(ts, error, label=f'$\Delta t = {delta_t:.1e}$, {toc:.3f} seconds')
    plt.ylabel('$||x_{\Delta t_i} - x_{10^{-5}}||$')
    plt.yscale('log')
    plt.xlabel('Time')
    plt.legend()
    plt.ylim([1e-5, 1e1])
    plt.savefig('c.3.png')
    plt.show()


def plot_dynamic_time_step(x0, p, u):
    delta_t = 1e-2
    t1 = 40
    subsample = int(np.round(1e5 * delta_t))
    subsample = np.s_[::subsample]
    kwargs = dict(
        x3=x0,
        p=p,
        u=u,
        t1=t1,
        delta_t=delta_t,
        factory=implicit.get_trapezoid_f,
        guess=explicit.forward_euler,
        dx_error_max=1e-4
    )
    tic = time.time()
    xs = list(tqdm(implicit.dynamic_step(**kwargs), total=int(t1 / delta_t)))[:]
    toc = time.time() - tic
    golden = np.load('xs.npy')
    error = np.linalg.norm(golden[subsample] - xs, np.inf, axis=1)
    ts = np.linspace(0, 40, error.shape[0])
    plt.plot(ts, error, label="Dynamic Time Step")
    plt.ylabel('$||x_{{dynamic}} - x_{{golden}}||$')
    plt.yscale('log')
    plt.xlabel('Time')
    plt.ylim([1e-5, 1e0])
    plt.title(f'{toc:.3f} seconds')
    plt.savefig('c.4.png')
    plt.show()


def plot_rk4_coarsening(x0, p, u):
    golden = np.load('xs.npy')
    for delta_t in np.geomspace(1e-1, 3e-1, 3):
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
        xs = list(tqdm(explicit.simulate(**kwargs), total=int(t1 / delta_t)))
        toc = time.time() - tic
        subsample = np.linspace(0, golden.shape[0]-1, len(xs)).astype(int)
        error = np.linalg.norm(golden[subsample] - xs, np.inf, axis=1)
        ts = np.linspace(0, 40, error.shape[0])
        plt.plot(ts, error, label=f'$\Delta t = {delta_t:.1e}$, {toc:.3f} seconds', alpha=0.5, dashes=[1, 1])
    plt.ylabel('$||x_{\Delta t_i} - x_{10^{-5}}||$')
    plt.yscale('log')
    plt.xlabel('Time')
    plt.legend()
    plt.ylim([1e-8, 1e0])
    plt.title('RK4')
    plt.savefig('c.5.png')
    plt.show()


if __name__ == '__main__':
    x0, p, u = generate_stochastic_inputs(3)
    if True: # comment this bit out once you find a good setting
        np.save('x0.npy', x0)
        np.save('p.npy', p)
    x0 = np.load('x0.npy')
    p = np.load('p.npy', allow_pickle=True)[()]
    plot_golden(x0, p, u)
    plot_forward_euler_coarsening(x0, p, u)
    plot_trapezoid_coarsening(x0, p, u)
    plot_dynamic_time_step(x0, p, u)
    plot_rk4_coarsening(x0, p, u)