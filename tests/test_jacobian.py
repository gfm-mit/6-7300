from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pytest

from domain_specific.evalf import evalf
from domain_specific.x0 import generate_inputs, generate_lognormal_input
from domain_specific.jacobian import finiteDifferenceJacobian, evalJacobian

@pytest.mark.xfail(reason="analytic and finite difference estimates of dmu/dy_tilde don't match", strict=True)
def test_finite_equals_analytic():
    x, p, u = generate_lognormal_input(3)
    print(x)
    print(p)
    print(u)

    J_finite = finiteDifferenceJacobian(evalf, x, p, u, 1e-15)
    print()
    J_analytical = evalJacobian(x, p, u)

    delta = np.linalg.norm(J_finite - J_analytical) 
    assert delta < 1e-7, delta
        
