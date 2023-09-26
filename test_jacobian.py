import numpy as np
from evalf import evalf
from test_evalf import generate_inputs
from jacobian import finiteDifferenceJacobian, evalJacobian

if __name__ == '__main__':
    x, p, u = generate_inputs(3)
    print(x)
    print(p)
    print(u)

    print(np.round(finiteDifferenceJacobian(evalf, x, p, u), 3))
    print()
    print(np.round(evalJacobian(x, p, u), 3))