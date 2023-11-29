from matplotlib import pyplot as plt
import numpy as np

from linear.jacobian_implicit import jf_product, gcr_implicit_wrapper
from linear.gcr import gcrWrapper
from domain_specific.evalf import evalf
from domain_specific.x0 import generate_deterministic_inputs, generate_stochastic_inputs
from domain_specific.jacobian import evalJacobian
from utils.performance import measure_speed, measure_mem, measure_eps_effect_gcr, measure_eps_effect_one_step


def test_jacobian_product():
    n = 2
    x, p, u = generate_deterministic_inputs(n)

    r = x.copy()
    r[1] = [0.5, 0.5]
    r = r.reshape(-1, )

    x = x.reshape(-1, )
    J = evalJacobian(x, p, u)

    Jr_explicit = J.dot(r)
    Jr_implicit = jf_product(x, p, u, r, eps=1e-10)
    assert np.linalg.norm(Jr_explicit - Jr_implicit) < 1e-6


def test_implicit_jacobian_gcr_delta_x():
    n = 2
    x0, p, u = generate_deterministic_inputs(n)
    r = x0.copy()
    r[1] = [0.5, 0.5]
    r = r.reshape(-1, )
    x0 = x0.reshape(-1, )

    x_implicit, r_norms = gcr_implicit_wrapper(x0=x0, p=p, u=u, tolrGCR=1e-4, eps=1e-10)
    x_explicit, r_norms, k = gcrWrapper(x0=x0, p=p, u=u, tolrGCR=1e-4)

    delta = np.linalg.norm(x_implicit - x_explicit)
    assert delta < 1e-4, delta

def test_implicit_jacobian_gcr_delta_f():
    n = 2
    x0, p, u = generate_stochastic_inputs(n)
    x0 = x0.reshape(-1, )

    x1, r_norms = gcr_implicit_wrapper(x0=x0, p=p, u=u, tolrGCR=1e-6, eps=1e-6)
    f = evalf(x0, t=None, p=p, u=u)
    J = evalJacobian(x0, p, u)

    rel_error = np.linalg.norm(f + J @ x1) / np.linalg.norm(f)
    assert rel_error < 1e-6


def test_measure_eps_effect_one_step():
    df, error = measure_eps_effect_one_step(np.geomspace(1e-10, 1e-2, 10), n=10)
    assert np.max(error) < 1e-4, error

def test_measure_eps_effect_gcr():
    # TODO: try this with explicit measure of error, not just relying on GCR report of residual
    # TODO: try this with larger n
    triples, b = measure_eps_effect_gcr([1e-10, 1e-2], n=2)
    assert np.max(triples.final) < 1e-4, triples.error

def test_measure_mem():
    f_size, J_size, f_peak, J_peak = measure_mem([3, 30])
    assert f_peak[0] > J_peak[0], "f should be larger than J at small n"
    assert f_peak[1] < J_peak[1], "f should be smaller than J at large n"

def test_measure_speed():
    f_time, J_time = measure_speed([10, 100])
    assert (np.array(f_time) < np.array(J_time)).all()