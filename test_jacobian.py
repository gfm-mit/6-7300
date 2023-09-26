import numpy as np
from evalf import evalf
from test_evalf import generate_inputs
from jacobian import finiteDifferenceJacobian, evalJacobian

if __name__ == '__main__':
    x, p, u = generate_inputs(3)
    print(x)
    print(p)
    print(u)

    print(finiteDifferenceJacobian(evalf, x, p, u, 1e-8))
    print()
    print(evalJacobian(x, p, u)) 