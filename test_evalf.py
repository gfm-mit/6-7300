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
    # y
    plt.plot(ans[:, 0], label="Node 1")
    plt.plot(ans[:, 1], label="Node 2")
    plt.plot(ans[:, 2], label="Node 3")
    plt.legend()
    plt.title("Evolution of true currency")
    plt.ylabel("True currency")
    plt.xlabel("Time")
    plt.show()

    # tilde_Y
    plt.plot(ans[:, 3], label="Node 1")
    plt.plot(ans[:, 4], label="Node 2")
    plt.plot(ans[:, 5], label="Node 3")
    plt.legend()
    plt.title("Evolution of effective currency")
    plt.ylabel("Effective currency")
    plt.xlabel("Time")
    plt.show()

    # mu
    plt.plot(ans[:, 6], label="Node 1")
    plt.plot(ans[:, 7], label="Node 2")
    plt.plot(ans[:, 8], label="Node 3")
    plt.legend()
    plt.title("Evolution of currency drift")
    plt.ylabel("Currency drift")
    plt.xlabel("Time")
    plt.show()

    return


if __name__ == '__main__':
    x0 = np.array([[1, 1, 1], [2, 2, 2], [0, 0, 0]]).reshape(9,)
    t = np.linspace(0, 10, 10)
    ans = test(x0, t)
    plot_evolution(ans[0])
