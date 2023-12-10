import sys
import os
import pathlib

import numpy as np
import pytest


sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))

from domain_specific.evalf import evalf
from domain_specific.x0 import generate_deterministic_inputs, generate_stochastic_inputs
from newton.from_julia import newton_julia_jacobian_free_wrapper, newton_julia_stepsize_wrapper


def test_simple_case_julia_stepsize():
    x0, p, u = generate_deterministic_inputs(3)

    x1 = newton_julia_stepsize_wrapper(x0, p, u)
    f = evalf(x1, t=None, p=p, u=u)

    error = np.linalg.norm(f, np.inf) 
    assert error < 1e-4, error

def test_simple_case_jacobian_free_julia():
    x0, p, u = generate_deterministic_inputs(3)

    x1 = newton_julia_jacobian_free_wrapper(x0, p, u)
    f = evalf(x1, t=None, p=p, u=u)

    error = np.linalg.norm(f) 
    assert error < 1e-4, error

def test_negative_currencies():
    for _ in range(10):
        x0, p, u = generate_stochastic_inputs(3)

        x1 = newton_julia_jacobian_free_wrapper(x0, p, u)

def test_100_countries():
    x0, p, u = generate_deterministic_inputs(100)

    x1 = newton_julia_jacobian_free_wrapper(x0, p, u)
    f = evalf(x1, t=None, p=p, u=u)

    error = np.linalg.norm(f) 
    assert error < 1e-4, error

def test_jacobian_versus_jacobian_free():
    x0, p, u = generate_deterministic_inputs(3)

    x1 = newton_julia_stepsize_wrapper(x0, p, u)
    x2 = newton_julia_jacobian_free_wrapper(x0, p, u)

    error = np.linalg.norm(x1 - x2)
    assert error < 1e-4, error