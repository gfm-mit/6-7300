from io import StringIO
import re
import numpy as np

# only for testing
import sys
import os
import pathlib

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
# only for testing

from domain_specific.evalf import evalf
from domain_specific.x0 import generate_inputs, generate_lognormal_input
from newton.from_matlab import newton_nd, newton_matlab_wrapper
from newton.from_julia import newton_julia_jacobian_free_wrapper, newton_julia_wrapper
from newton.homotopy import continuation_taylor0, continuation_taylor1, continuation_taylor2, newton_continuation_wrapper

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

def test_continuation1():
    x0, p, u = generate_lognormal_input(3)
    x3 = np.reshape(x0, [-1]).copy()

    x1 = newton_continuation_wrapper(
        x0, p, u,
        qs=np.linspace(0, 1, 3, endpoint=True),
        fqs=continuation_taylor1)
    f = evalf(x1, t=None, p=p, u=u)

    error = np.linalg.norm(f) 
    assert error < 1e-4, error