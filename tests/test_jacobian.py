from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pytest
# only for testing
import sys
import os
import pathlib

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))

from domain_specific.evalf import evalf
from domain_specific.x0 import generate_deterministic_inputs, generate_stochastic_inputs
from domain_specific.jacobian import finiteDifferenceJacobian, evalJacobian

def test_finite_equals_analytic():
    x, p, u = generate_stochastic_inputs(3)
    print(x)
    print(p)
    print(u)

    D = 1e-7 # this actually WAS tested carefully, and looks like the right result
    J_finite = finiteDifferenceJacobian(evalf, x, p, u, D)
    print()
    J_analytical = evalJacobian(x, p, u)

    delta = np.linalg.norm(J_finite - J_analytical) 
    assert delta < 1e-5, delta

def test_many_random_inputs():
    for _ in range(100):
        test_finite_equals_analytic()

