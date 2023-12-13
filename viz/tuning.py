import einops
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import sys
import os
import pathlib

from tqdm import tqdm

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
from domain_specific.jacobian import evalJacobian, finiteDifferenceJacobian
from domain_specific.x0 import generate_demo_inputs
from dynamic import explicit
from domain_specific.evalf import evalf
import newton.from_julia


def plot_spectrum(u, p_initial, x1):
    J1 = finiteDifferenceJacobian(evalf, x1, p_initial, u)
    idx = np.arange(J1.shape[0])
    J1 = J1[idx % 10 < 4]
    J1 = J1[:, idx % 10 < 4]
    lambda1 = np.linalg.eigvals(J1)
    lambda1 = (10 + np.log(lambda1)).real * np.exp(1j*np.log(lambda1).imag)
    plt.scatter(lambda1.real, lambda1.imag, alpha=0.5)
    center_spines()


def center_spines():
    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def test_spectrum(x0, p, u, k, v, axs):
    plt.sca(axs[0])

    p_initial = p.copy()
    p_initial['d'] = p_initial['d'][0, :, :]
    x1 = newton.from_julia.newton_julia_jacobian_free_wrapper(x0, p_initial, u)

    p_final = p.copy()
    p_final['d'] = p_final['d'][-1, :, :]
    plot_spectrum(u, p_final, x1)
    x2 = newton.from_julia.newton_julia_jacobian_free_wrapper(x1, p_final, u)

    p_test = p.copy()
    p_test['d'] = p_test['d'][-1, :, :]
    if k == "BACKWARDS":
        p_test = p_initial.copy()
    elif k == "d":
        p_test['d'][2, 0] *= v
        p_test['d'][0, 2] *= v
    else:
        p_test[k] = v
    x3 = newton.from_julia.newton_julia_jacobian_free_wrapper(x1, p_test, u)
    x4 = explicit.rk4(x1, p_test, u, 1e-6)
    plot_spectrum(u, p_test, x1)

    plt.xlabel("real")
    plt.ylabel("imag")
    plt.title(k)

    plt.sca(axs[1])
    delta = x3 - x2
    delta = np.round(delta[:4], 3)
    plt.scatter(delta[:4], np.arange(4), color="plum")
    f2 = evalf(x1, None, p_final, u)
    f3 = evalf(x1, None, p_test, u)
    delta_f = (f3-f2)[20:24]
    plt.scatter(1e-2*delta_f, np.arange(4), color="limegreen")
    for i in range(4):
        plt.text(delta[i], i, "US EU IN CN".split()[i])
    plt.xscale('symlog', linthresh=1e-2)
    center_spines()


def multi_spectrum(x0, p, u):
    epsilon = np.exp(1)
    fig, axs = plt.subplots(2, 6)
    axs = axs.transpose()
    test_spectrum(x0, p, u, "tau1", p['tau1'] * epsilon, axs[1])
    test_spectrum(x0, p, u, "tau2", p['tau2'] * epsilon, axs[2])
    test_spectrum(x0, p, u, "tau3", p['tau3'] * epsilon, axs[3])
    test_spectrum(x0, p, u, "alpha", p['alpha'] * epsilon, axs[4])
    test_spectrum(x0, p, u, "d", epsilon, axs[5])
    test_spectrum(x0, p, u, "BACKWARDS", epsilon, axs[0])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Generate data to visualize
    x0, p, u = generate_demo_inputs(10)
    #np.random.seed()
    multi_spectrum(x0, p, u)