from scipy.integrate import odeint
import numpy as np
from evalf import evalf, generate_inputs, get_E
import matplotlib.pyplot as plt


def test(x0, t, n=3):
    E = get_E('configs/test.txt')
    x, p, u = generate_inputs(n, E)
    ans = odeint(evalf, x0, t, args=(p, u, E), full_output=True)
    return ans


def plot_evolution(ans, n=3):
    """
    [[y_1, y_2, y_3, ...],
    [tilde_y_1, tilde_y_2, tilde_y_3, ...],
    [mu_1, mu_2, mu_3, ...],
    ...]
    """
    print(ans.shape)
    # Iterating over each timestep
    y_evolution, tilde_y_evolution, mu_evolution = [], [], []
    """
    for t in ans.shape[0]:
        for node in range(n):
            y_evolution.append()
            tilde_y_evolution.append()
            mu_evolution.append()
    """
    return


if __name__ == '__main__':
    x0 = np.array([[1, 1, 1], [2, 2, 2], [0, 0, 0]]).reshape(9,)
    t = np.linspace(0, 10, 10)
    ans = test(x0, t)
    plot_evolution(ans[0])
