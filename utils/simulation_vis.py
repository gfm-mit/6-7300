import numpy as np
import matplotlib.pyplot as plt
import einops
from tqdm import tqdm

from domain_specific import evalf, jacobian

def visualize(xs, p, u, savefig=None, t=None):
    n = xs.shape[1] // 3
    stacked = einops.rearrange(xs, 't (d c) -> c d t', d=3)
    x_norms = np.linalg.norm(xs, axis=1)
    F = list(tqdm([evalf.evalf(x, None, p, u) for x in xs]))
    F_cond = list(tqdm([np.linalg.norm(f) for f in F]))
    J = list(tqdm([jacobian.finiteDifferenceJacobian(evalf.evalf, x, p, u) for x in xs]))
    J_cond = list(tqdm([np.linalg.cond(j) for j in J]))
    # TODO: also somehow measure dissipativity (whether eigenspectrum is all negative)
    fig, axs = plt.subplots(3)
    if t is None:
        t = stacked.shape[2]
    ts = np.linspace(0, t, stacked.shape[2])
    for i in range(n):
        plt.sca(axs[0])
        color = plt.plot(ts, stacked[i, 0], label=f"Country {i}")[0].get_color()
        plt.plot(ts, stacked[i, 1], color=color, dashes=[1,1])
        plt.sca(axs[1])
        plt.plot(ts, stacked[i, 2], color=color, alpha=0.25, linewidth=3, zorder=-10)
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
    plt.plot(ts, x_norms, color="turquoise", label="$|x|_2$")
    plt.plot(ts, F_cond, color="cornflowerblue", label="$|f|$")
    plt.plot(ts, J_cond, color="black", label="$|J||J^{-1}|$")
    plt.legend()
    plt.xlabel('time')
    plt.yscale('log')
    plt.ylabel('norm')
    plt.legend()
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()

def eigenval_plot(x, p, u):
    vals = np.linalg.eigvals(jacobian.finiteDifferenceJacobian(evalf.evalf, x, p, u))
    fig, ax = plt.subplots()
    ax.scatter(np.real(vals), np.imag(vals), marker='x', s=100, color="orange")  # Fixed 'shape' to 'marker'
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.xlabel('real')
    plt.ylabel('imag')
    plt.title('eigenvalues of Jacobian')
    plt.xscale('symlog', linthresh=1e-3)
    plt.yscale('symlog', linthresh=1e-3)
    plt.show()
    assert False, (vals)