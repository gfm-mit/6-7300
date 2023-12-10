import numpy as np
import matplotlib.pyplot as plt


def plot_evolution(ans, n=3):
    """
    [[y_1, y_2, y_3, ...],
    [tilde_y_1, tilde_y_2, tilde_y_3, ...],
    [mu_1, mu_2, mu_3, ...],
    ...]
    """
    fig, ax = plt.subplots(ans.shape[0], ans.shape[1], figsize=(8, 8))
    for row in range(ans.shape[0]):
        for col in range(ans.shape[1]):
            ax[row][col].plot(ans[row, col], label="Node {}".format(col))
            ax[0][col].set_title("Country {}".format(col))
            ax[row][col].axhline(y=0, color="lightgray", dashes=[1,1], zorder=-10)
            ax[row][col].set_ylim([-100, 100])
            ax[row][col].set_yscale('symlog', linthresh=1e-2)

    ax[0][0].set_ylabel("True currency")
    ax[0][0].set_xlabel("Time")

    ax[1][0].set_ylabel("Effective currency")
    ax[1][0].set_xlabel("Time")

    ax[2][0].set_ylabel("Currency drift")
    ax[2][0].set_xlabel("Time")
    plt.tight_layout()
    plt.show()
    return


def plot_preconditioned_eigenvals(P, J):
    K0 = np.round(sorted(np.log10(np.abs(np.linalg.eigvals(J)))), 1)
    K1 = np.round(sorted(np.log10(np.abs(np.linalg.eigvals(P @ J)))), 1)
    plt.hist(K0, bins=20)
    plt.hist(K1, bins=20, alpha=0.5)
    plt.show()
    assert False, np.stack([K0, K1], axis=1)