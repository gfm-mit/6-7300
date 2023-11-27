import numpy as np
import matplotlib.pyplot as plt
import einops
from tqdm import tqdm

from domain_specific import evalf, jacobian

def visualize(xs, p, u):
    stacked = einops.rearrange(xs, 't (d c) -> c d t', d=3)
    x_norms = np.linalg.norm(xs, axis=1)
    F = list(tqdm([evalf.evalf(x, None, p, u) for x in xs]))
    F_cond = list(tqdm([np.linalg.norm(f) for f in F]))
    J = list(tqdm([jacobian.finiteDifferenceJacobian(evalf.evalf, x, p, u) for x in xs]))
    J_cond = list(tqdm([np.linalg.cond(j) for j in J]))
    fig, axs = plt.subplots(3)
    for i in range(3):
        plt.sca(axs[0])
        color = plt.plot(stacked[i, 0], label=f"Country {i}")[0].get_color()
        plt.plot(stacked[i, 1], color=color, dashes=[1,1])
        plt.sca(axs[1])
        plt.plot(stacked[i, 2], color=color, alpha=0.25, linewidth=3, zorder=-10)
    plt.sca(axs[0])
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('currency value')
    plt.yscale('symlog')
    plt.title('Solid is $y$, dashed lines are $\~{y}$')
    plt.sca(axs[1])
    plt.xlabel('time')
    plt.ylabel('$\Delta$ currency value')
    plt.yscale('symlog')
    plt.title('$\mu$')
    plt.sca(axs[2])
    plt.title('Simulation Stats')
    plt.plot(x_norms, color="turquoise", label="$|x|_2$")
    plt.plot(F_cond, color="cornflowerblue", label="$|f|$")
    plt.plot(J_cond, color="black", label="$|J||J^{-1}|$")
    plt.legend()
    plt.xlabel('time')
    plt.yscale('log')
    plt.ylabel('norm')
    plt.legend()
    plt.tight_layout()
    plt.show()
