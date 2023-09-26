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
            else:
                ax[row][col].set_ylim([-drift, drift])

    ax[0][0].set_ylabel("True currency")
    ax[0][0].set_xlabel("Time")

    ax[1][0].set_ylabel("Effective currency")
    ax[1][0].set_xlabel("Time")

    ax[2][0].set_ylabel("Currency drift")
    ax[2][0].set_xlabel("Time")
    plt.tight_layout()
    plt.show()
    return