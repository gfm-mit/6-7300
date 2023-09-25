from scipy.integrate import odeint
import numpy as np
from evalf import evalf, generate_inputs, get_E
import matplotlib.pyplot as plt
import einops


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
    fig, ax = plt.subplots(ans.shape[0], ans.shape[1], figsize=(8, 8))
    drift = np.max(np.abs(ans[2]))
    max_y = np.max(ans[:1])
    min_y = np.min(ans[:1])
    for row in range(ans.shape[0]):
        for col in range(ans.shape[1]):
            if row < 2:
                ax[row][col].set_yscale('log')
            ax[row][col].plot(ans[row, col], label="Node {}".format(col))
            ax[0][col].set_title("Country {}".format(col))
            ax[row][col].axhline(y=0, color="lightgray", dashes=[1,1], zorder=-10)
            if row < 2:
                ax[row][col].set_ylim([min_y, max_y])
            elif row == 2:
                ax[row][col].set_ylim([-drift, drift])
            else:
                ax[row][col].set_ylim([-.1, .1])

    ax[0][0].set_ylabel("True currency")
    ax[0][0].set_xlabel("Time")

    ax[1][0].set_ylabel("Effective currency")
    ax[1][0].set_xlabel("Time")

    ax[2][0].set_ylabel("Currency drift")
    ax[2][0].set_xlabel("Time")
    plt.tight_layout()
    plt.show()

    return


# test cases, all with two countries:
# 1) all equal
# 2) all equal, but y_tilde starts too high
# 3) start with slightly different currency values, should converge
# 4) test with very small tau2 / tau_mu
if __name__ == '__main__':
    T = 20
    x0 = np.array([
        [1, 1.1], 
        [1, 1.1],
        [0, 0],
        [0, 0] # meaningless
        ]).reshape(-1,)
    t = np.linspace(0, T, T)
    E = get_E("configs/test.txt")
    x, p, u = generate_inputs(2, E)
    F = evalf(x0, t, p, u, E, debug=True)
    F = np.reshape(F, [4, -1]).transpose()
    ans = test(x0, t)[0]
    ans = einops.rearrange(ans, "t (d n) -> d n t", d=4)
    print(F)
    #print(ans)
    plot_evolution(ans)
