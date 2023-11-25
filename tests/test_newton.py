import sys
import os
import pathlib

import numpy as np


sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))

from domain_specific.evalf import evalf
from domain_specific.jacobian import evalJacobian
from domain_specific.x0 import generate_inputs, generate_lognormal_input
from newton.from_matlab import newton_matlab_wrapper
from newton.from_julia import newton_julia_jacobian_free_wrapper, newton_julia_wrapper
from newton.homotopy import standard, taylor, diag, newton_continuation_wrapper


def test_simple_case_converges():
    x0, p, u = generate_lognormal_input(3)

    x1 = newton_matlab_wrapper(x0, p, u)
    f = evalf(x1, t=None, p=p, u=u)

    error = np.linalg.norm(f) 
    assert error < 1e-4, f

def test_simple_case_julia():
    x0, p, u = generate_lognormal_input(3)

    x1 = newton_julia_wrapper(x0, p, u)
    f = evalf(x1, t=None, p=p, u=u)

    error = np.linalg.norm(f) 
    assert error < 1e-4, error

def test_simple_case_jacobian_free_julia():
    x0, p, u = generate_inputs(3)

    x1 = newton_julia_jacobian_free_wrapper(x0, p, u)
    f = evalf(x1, t=None, p=p, u=u)

    error = np.linalg.norm(f) 
    assert error < 1e-4, error

def test_continuation():
    x0, p, u = generate_inputs(3)

    x1 = newton_continuation_wrapper(
        x0, p, u,
        qs=[0, 0.5, 1],
        fqs=diag)
    f = evalf(x1, t=None, p=p, u=u)

    error = np.linalg.norm(f) 
    assert error < 1e-4, error