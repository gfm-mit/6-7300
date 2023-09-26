import numpy as np
from evalf import evalf
from test_evalf import generate_inputs
from jacobian import finiteDifferenceJacobian, evalJacobian


def test_structure():
    x, p, u = generate_inputs(3)
    print(x)
    print(p)
    print(u)

    J_finite = finiteDifferenceJacobian(evalf, x, p, u, 1e-8)
    print()
    J_analytical = evalJacobian(x, p, u)

    assert(J_finite == J_analytical)

