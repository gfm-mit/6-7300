import sys
import os
import pathlib

import numpy as np
import pytest


sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))

from domain_specific.evalf import evalf
from domain_specific.jacobian import evalJacobian
from domain_specific.x0 import generate_inputs, generate_lognormal_input
from newton.from_julia import newton_julia_jacobian_free_wrapper, newton_julia_stepsize_wrapper, newton_julia_wrapper
from newton.homotopy import alpha, mu_only, standard, taylor, diag, newton_continuation_wrapper


def test_simple_case_julia_stepsize():
    x0, p, u = generate_inputs(3)

    x1 = newton_julia_stepsize_wrapper(x0, p, u)
    f = evalf(x1, t=None, p=p, u=u)

    error = np.linalg.norm(f, np.inf) 
    assert error < 1e-4, error

def test_simple_case_jacobian_free_julia():
    x0, p, u = generate_inputs(3)

    x1 = newton_julia_jacobian_free_wrapper(x0, p, u)
    f = evalf(x1, t=None, p=p, u=u)

    error = np.linalg.norm(f) 
    assert error < 1e-4, error

def test_negative_currencies():
    for _ in range(10):
        x0, p, u = generate_lognormal_input(3)

        x1 = newton_julia_jacobian_free_wrapper(x0, p, u)

def test_100_countries():
    x0, p, u = generate_inputs(100)

    x1 = newton_julia_jacobian_free_wrapper(x0, p, u)
    f = evalf(x1, t=None, p=p, u=u)

    error = np.linalg.norm(f) 
    assert error < 1e-4, error

def test_jacobian_versus_jacobian_free():
    x0, p, u = generate_inputs(3)

    x1 = newton_julia_stepsize_wrapper(x0, p, u)
    x2 = newton_julia_jacobian_free_wrapper(x0, p, u)

    error = np.linalg.norm(x1 - x2)
    assert error < 1e-4, error

@pytest.mark.skip("continuation hasn't worked so far, in the old version either")
def test_continuation():
    x0, p, u = generate_lognormal_input(3)

    x1 = newton_continuation_wrapper(
        x0, p, u,
        qs=[1],
        fqs=standard)
    f = evalf(x1, t=None, p=p, u=u)

    error = np.linalg.norm(f) 
    assert error < 1e-4, error