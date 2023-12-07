from matplotlib import pyplot as plt
import pandas as pd
import scipy
from scipy.integrate import odeint
import numpy as np
import einops
import sdeint
import time
# only for testing
import sys
import os
import pathlib

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))

from domain_specific.evalf import evalf, evalg, get_exports, get_exports_for_loops
from domain_specific.jacobian import evalJacobian
from domain_specific.x0 import generate_deterministic_inputs, generate_stochastic_inputs
from utils import simulation_vis
from utils.plot_util import plot_evolution


def test_get_exports_for_loops():
    n = 100
    x, p, u = generate_stochastic_inputs(n)
    y_tilde = x[1]
    x_for = get_exports_for_loops(y_tilde, p)
    x_np = get_exports(y_tilde, p)
    assert (x_for.shape == x_np.shape), (x_for.shape, x_np.shape)
    assert (np.abs(x_for - x_np) < 1e-15).all(), np.max(np.abs(x_for - x_np))


def test_timing():
    n = 100
    x, p, u = generate_stochastic_inputs(n)
    y_tilde = x[1]
    tic = time.time()
    for _ in range(int(1e2)):
        x_for = get_exports_for_loops(y_tilde, p)
    toc = time.time()
    slow = toc - tic
    tic = time.time()
    for _ in range(int(1e2)):
        x_for = get_exports(y_tilde, p)
    toc = time.time()
    fast = toc - tic
    assert fast < 1e-2 * slow
    print(f"for loops: {toc - tic}seconds")
    print(f"numpy: {toc - tic}seconds")


def test_symmetric_equilibrium():
    x, p, u = generate_deterministic_inputs(2)
    x0 = np.array([
        [0, 0], 
        [0, 0],
        [0, 0]
        ]).reshape(-1,)
    dx = evalf(x0, None, p, u)
    dx = einops.rearrange(dx, "(n d) -> d n", d=2)
    assert (dx == 0).all()


def test_convergence():
    T = 1000
    x, p, u = generate_stochastic_inputs(2)
    p['sigma'] = np.zeros([2])
    x0 = np.array([
        [0, 0.01],
        [0, 0.01],
        [0, 0]
        ]).reshape(-1,)
    t = np.linspace(0, T, T)
    def f_wrapper(x, t):
        return evalf(x, t, p, u)
    def g_wrapper(x, t):
        g = evalg(x, t, p, u)[:]
        return g
    ans = sdeint.itoint(f_wrapper, g_wrapper, x0, t)
    ans_plot = einops.rearrange(ans, "t (d n) -> d n t", d=3)
    #plot_evolution(ans_plot)

    ans = ans[999].reshape(3, 2)
    n1_ans = ans[:, 0]
    n2_ans = ans[:, 1]
    assert(round(n1_ans[0], 13) == round(n2_ans[0], 13))    # Justify rounding with condition number
    assert(round(n1_ans[1], 13) == round(n2_ans[1], 13))    # Noise at 2 decimal points


def test_delays():
    T = 100
    # Small time delay should converge more quickly
    x, p, u = generate_deterministic_inputs(2)
    #p['sigma'] = np.zeros([2])
    p['tau1'] = 1
    x0 = np.array([
        [0, .1],
        [0, .1],
        [0, 0]
    ]).reshape(-1, )
    t = np.linspace(0, T, T)
    def f_wrapper(x, t):
        return evalf(x, t, p, u)
    def g_wrapper(x, t):
        g = evalg(x, t, p, u)[:]
        return g
    ans_sm = sdeint.itoint(f_wrapper, g_wrapper, x0, t)
    ans_sm = einops.rearrange(ans_sm, "t (d n) -> d n t", d=3)
    #plot_evolution(ans_sm)
    avg_sm_oscillation = measure_oscillations(ans_sm)

    # Large time delay should converge more slowly
    x, p, u = generate_deterministic_inputs(2)
    p['tau1'] = 10
    x0 = np.array([
        [0, .1],
        [0, .1],
        [0, 0]
    ]).reshape(-1, )
    t = np.linspace(0, T, T)
    ans_lg = sdeint.itoint(f_wrapper, g_wrapper, x0, t)
    ans_lg = einops.rearrange(ans_lg, "t (d n) -> d n t", d=3)
    #plot_evolution(ans_lg)
    avg_lg_oscillation = measure_oscillations(ans_lg)
    assert(np.mean(avg_sm_oscillation) < np.mean(avg_lg_oscillation))


def test_elasticity():
    T = 100
    x, p, u = generate_deterministic_inputs(2)

    # Low elasticity means less oscillations
    p['alpha'] = .1
    x0 = np.array([
        [0, .1],
        [0, .1],
        [0, 0]
        ]).reshape(-1,)
    t = np.linspace(0, T, T)
    def f_wrapper(x, t):
        return evalf(x, t, p, u)
    def g_wrapper(x, t):
        g = evalg(x, t, p, u)[:]
        return g
    ans = sdeint.itoint(f_wrapper, g_wrapper, x0, t)
    ans_low = einops.rearrange(ans, "t (d n) -> d n t", d=3)
    #plot_evolution(ans_low)
    avg_low_oscillation = measure_oscillations(ans_low)

    # High elasticity means more oscillations
    p['alpha'] = .5
    x0 = np.array([
        [0, .1],
        [0, .1],
        [0, 0]
    ]).reshape(-1, )
    t = np.linspace(0, T, T)
    ans = sdeint.itoint(f_wrapper, g_wrapper, x0, t)
    ans_high = einops.rearrange(ans, "t (d n) -> d n t", d=3)
    #plot_evolution(ans_high)
    avg_high_oscillation = measure_oscillations(ans_high)
    assert(np.mean(avg_low_oscillation) < np.mean(avg_high_oscillation))


def measure_oscillations(ans, start=80):
    smooth = scipy.ndimage.convolve1d(ans, [1/10]*10, axis=2, mode='constant')
    oscillation = np.mean(np.abs(ans - smooth))
    return oscillation


def runode(x0, t, p, u):
    ans = odeint(evalf, x0, t, args=(p, u), full_output=True)
    return ans

  
# test cases, all with two countries:
# 1) all equal (GF)
# 2) all equal, but y_tilde starts too high
# 3) start with slightly different currency values, should converge (JR)
# 4) test with very small tau2 / tau_mu (JR)
# 5) with large alpha, shouldn't ring (JR)
# 6) with small alpha, should ring (JR)
if __name__ == '__main__':
    test_convergence()
    if False:
        T = 100
        x0 = np.array([
            [1, 1.1], 
            [1, 1.1],
            [0, 0]
            ]).reshape(-1,)
        t = np.linspace(0, T, T)
        x, p, u = generate_deterministic_inputs(2)
        F = evalf(x0, t, p, u)
        F = np.reshape(F, [3, -1]).transpose()
        #ans = runode(x0, t)[0]
        def f_wrapper(x, t):
            return evalf(x, t, p, u)
        def g_wrapper(x, t):
            g = evalg(x, t, p, u)[:]
            return g
        G = evalf(x0, t, p, u)
        ans = sdeint.itoint(f_wrapper, g_wrapper, x0, t)
        ans = einops.rearrange(ans, "t (d n) -> d n t", d=3)
        #print(F)
        #print(ans)
        plot_evolution(ans)

