from io import StringIO
import re
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import einops
import pytest

# only for testing
import sys
import os
import pathlib

from utils.plot_util import plot_evolution

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
# only for testing

from domain_specific.x0 import generate_inputs, generate_lognormal_input
from domain_specific.evalf import evalf
from newton.from_julia import newton_julia_wrapper
from dynamic.explicit import rk4
import dynamic.explicit as explicit
import dynamic.implicit as implicit
from domain_specific.jacobian import evalJacobian

# this takes ones of minutes to run
def onetime_setup():
    print("generating golden data instead of running tests".format(__file__))
    x0, p, u = generate_inputs(3)
    delta_t = 1e-5

    tic = time.time()
    xs = [x for x in tqdm(explicit.simulate(x0, p, u, 20, delta_t))]
    toc = time.time()
    xs = np.stack(xs)[::100]
    np.save('tests/dynamic_golden_1e-3.npy', xs)
    print(toc - tic)


def test_equilibrium():
    x0, p, u = generate_inputs(3)
    delta_t = 1e-3

    xs = list(explicit.simulate(x0, p, u, 20, delta_t))
    x1 = xs[-1]
    J1 = evalJacobian(x1, p, u)
    print(np.linalg.cond(J1))
    x2 = newton_julia_wrapper(x1, p, u)
    error = np.linalg.norm(x2 - x1, np.inf)
    assert error < 1e-6, error

def test_forward_euler():
    x0, p, u = generate_inputs(3)
    delta_t = 1e-3

    xs = list(explicit.simulate(x0, p, u, 20, delta_t))
    xs = np.stack(xs)
    golden = np.load('tests/dynamic_golden_1e-3.npy')
    error = xs - golden
    error = np.linalg.norm(error, axis=1, ord=np.inf)

    assert (error < 1e-3).all(), error

@pytest.mark.xfail(reason="new parameterization is too stable")
def test_forward_euler_unstable():
    x0, p, u = generate_inputs(3)
    delta_t = 1e1

    xs = list(explicit.simulate(x0, p, u, 20, delta_t))
    xs = np.stack(xs)
    golden = np.load('tests/dynamic_golden_1e-3.npy')[::10000]
    error = xs - golden
    error = np.linalg.norm(error, axis=1, ord=np.inf)[1:] # there's no error on the first step, of course!

    # TODO, this is dumb!
    assert (error > 1).all(), error


def test_rk4():
    x0, p, u = generate_inputs(3)
    delta_t = 1e-1 # damn, son

    xs = list(explicit.simulate(x0, p, u, 20, delta_t, f_step=rk4))
    xs = np.stack(xs)
    golden = np.load('tests/dynamic_golden_1e-3.npy')[::100]
    error = xs - golden
    error = np.linalg.norm(error, axis=1, ord=np.inf)

    assert (error < 1e-5).all(), error


def test_backward_euler():
    x0, p, u = generate_inputs(3)
    delta_t = 1e-2

    xs = list(implicit.simulate(x0, p, u, 20, delta_t))
    xs = np.stack(xs)
    golden = np.load('tests/dynamic_golden_1e-3.npy')[::10]
    error = xs - golden
    error = np.linalg.norm(error, axis=1, ord=np.inf)
    error = np.max(error)
    assert error < 1e-2, error
    # embarrassing, but this doesn't work very well
    #assert error > 1e-4, error


def test_trapezoid():
    x0, p, u = generate_inputs(3)
    delta_t = 1e-2

    xs = list(implicit.simulate(x0, p, u, 20, delta_t, factory=implicit.get_trapezoid_f))
    xs = np.stack(xs)
    golden = np.load('tests/dynamic_golden_1e-3.npy')[::10]
    error = xs - golden
    error = np.linalg.norm(error, axis=1, ord=np.inf)
    assert (error < 1e-4).all(), error


# TODO: speedup is extremely mild
def test_dynamic_step(plot=True):
    x0, p, u = generate_lognormal_input(3)
    delta_t = 1e-1

    xs = list(explicit.simulate(x0, p, u, 10, delta_t, f_step=explicit.rk4))
    xs = np.stack(xs)
    #golden = np.load('tests/dynamic_golden_1e-3.npy')
    #error = xs - golden
    #error = np.linalg.norm(error, axis=1, ord=np.inf)
    xs = einops.rearrange(xs, "t (d n) -> d n t", d=3)
    #plot_evolution(xs)
    #assert (error < 1e-4).all(), error

if __name__ == "__main__":
    #onetime_setup()
    x0, p, u = generate_lognormal_input(3)
    delta_t = 1e-2 # damn, son
    golden = list(implicit.dynamic_step(x0, p, u, 100, delta_t, factory=implicit.get_trapezoid_f, guess=explicit.rk4, dx_error_max=1e-4))
    #golden = np.load('tests/dynamic_golden_1e-3.npy')
    golden = einops.rearrange(golden, 't (d c) -> c d t', d=3)
    for i in range(3):
        color = plt.plot(golden[i, 0], label=f"Country {i}")[0].get_color()
        plt.plot(golden[i, 1], color=color, dashes=[1,1])
        plt.plot(golden[i, 2], color=color, alpha=0.25, linewidth=3, zorder=-10)
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('THING')
    plt.yscale('symlog')
    plt.tight_layout()
    plt.show()
