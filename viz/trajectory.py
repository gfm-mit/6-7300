import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

import sys
import os
import pathlib

from tqdm import tqdm

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
from dynamic import explicit
import domain_specific.demo
import newton.from_julia


def plot_trajectory(p, u, x0):
    p_initial = p.copy()
    p_initial['d'] = p_initial['d'][0, :, :]
    x1 = newton.from_julia.newton_julia_jacobian_free_wrapper(x0, p_initial, u)
    t1 = 100
    kwargs = dict(
        x0=x1,
        p=p,
        u=u,
        t1=t1,
        delta_t=1e-2,
        f_step=explicit.rk4,
        demo=True
    )
    # NOTE: the shock is a drop in trade frictions between the two countries
    xs = np.array(list(explicit.simulate(**kwargs)))
    xms = np.stack(list(explicit.simulate_exports(xs, **kwargs)))
    xs = np.stack(xs)
    ts = np.linspace(0, t1, xs.shape[0])
    fig, axs = plt.subplots(3)
    for k, v in dict(
        US=0,
        EU=1,
        IN=2,
        CN=3,
    ).items():
        offset = -xs[0, v]
        plt.sca(axs[0])
        color = plt.plot(ts, offset + xs[:, v], label=k)[0].get_color()
        if v != 0:
            plt.sca(axs[1])
            xm_us = np.log(xms[:, 0, v])
            xm_us[:10] = np.nan
            xm_us = xm_us - xm_us[10]
            plt.plot(ts, xm_us, color=color, dashes=[1,1])
        if v != 2:
            plt.sca(axs[2])
            xm_in = np.log(xms[:, 2, v])
            xm_in[:10] = np.nan
            xm_in = xm_in - xm_in[10]
            plt.plot(ts, xm_in, color=color, dashes=[1,1])

    plt.sca(axs[0])
    plt.title("Currency Values")
    plt.xlabel("days since shock")
    plt.ylabel("log change")
    plt.legend()
    plt.sca(axs[1])
    plt.title("US Export Volume")
    plt.xlabel("days since shock")
    plt.ylabel("log change")
    plt.sca(axs[2])
    plt.title("IN Export Volume")
    plt.xlabel("days since shock")
    plt.ylabel("log change")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Generate data to visualize
    x0, p, u = domain_specific.demo.generate_wobble_inputs(10)
    np.random.seed()
    plot_trajectory(p, u, x0)