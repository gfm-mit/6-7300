from scipy.integrate import odeint
import numpy as np
from evalf import evalf, generate_inputs, get_E


def test(x0, t):
    E = get_E('configs/test.txt')
    x, p, u = generate_inputs(3, E)
    odeint(evalf, x0, t, args=(p, u, E), full_output=True)
    return


if __name__ == '__main__':
    x0 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]).reshape(9,)
    t = np.linspace(0, 10, 10)
    test(x0, t)

