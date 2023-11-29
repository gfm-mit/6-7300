from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pytest

from domain_specific.evalf import evalf
from domain_specific.x0 import generate_inputs, generate_lognormal_input
from domain_specific.jacobian import finiteDifferenceJacobian, evalJacobian

def test_finite_equals_analytic():
    x, p, u = generate_lognormal_input(3)
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

