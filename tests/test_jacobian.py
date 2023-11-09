import numpy as np
from domain_specific.evalf import evalf
from domain_specific.x0 import generate_inputs
from domain_specific.jacobian import finiteDifferenceJacobian, evalJacobian


def test_finite_equals_analytic():
    x, p, u = generate_inputs(3)
    print(x)
    print(p)
    print(u)

    J_finite = finiteDifferenceJacobian(evalf, x, p, u, 1e-8)
    print()
    J_analytical = evalJacobian(x, p, u)

    delta = np.linalg.norm(J_finite - J_analytical) 
    assert delta < 1e-7, delta
    
