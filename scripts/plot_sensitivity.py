from domain_specific.x0 import generate_stochastic_inputs
import matplotlib.pyplot as plt
import einops
import numpy as np
from utils import sensitivity
import seaborn as sns


def visualize(xs, xs_perturb, savefig=None):
    stacked = einops.rearrange(xs, 't (d c) -> c d t', d=3)
    # x_norms = np.linalg.norm(xs, axis=1)
    stacked_perturb = einops.rearrange(xs_perturb, 't (d c) -> c d t', d=3)
    # x_norms_perturb = np.linalg.norm(xs_perturb, axis=1)
    for i in range(3):
        color = plt.plot(stacked[i, 0], label=f"Country {i}")[0].get_color()
        plt.plot(stacked[i, 1], color=color, dashes=[1, 1])
        plt.plot(stacked_perturb[i, 1], color=color, alpha=0.25, linewidth=3, zorder=-10)
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('currency value')
    plt.yscale('symlog')
    plt.title('Solid is $y$, dashed lines are $y_p$')
    plt.legend()
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    plt.clf()


if __name__ == "__main__":
    p_key = 'tau1'
    t1 = 40
    data = {'eps': [], 'diff': [], 'state': []}
    eps = [10e-8, 10e-10, 10e-12, 10e-14, 10e-16]
    for state in ['y', r'\tilde{y}', r'\mu']:
        avgs = []
        for dp in eps:
            diffs = []
            i, tries = 0, 0
            while i < 5 and tries < 3:
                try:
                    x0, p, u = generate_stochastic_inputs(3, stochastic=[state])
                    xs, xs_perturb = sensitivity.analyze_sensitivity(x0, p, u, p_key, dp, t1=t1)
                    diffs.append(max(np.abs(np.subtract(xs_perturb[:, 1], xs[:, 1]) / dp)))
                    data['diff'].append(max(np.abs(np.subtract(xs_perturb[:, 1], xs[:, 1]) / dp)))
                    data['state'].append('$%s$' % state)
                    data['eps'].append(dp)
                    i += 1
                    # visualize(xs, xs_perturb)
                except:
                    print("Exception!")
                    tries += 1
                    pass
            print("Done!")
            avgs.append(np.mean(diffs))
        plt.plot(eps, avgs)

    plt.xscale("log")
    p1 = sns.scatterplot(data=data, x='eps', y='diff', hue='state',
                       alpha=.2, legend=True,
                       )
    p1.set_xscale("log")
    plt.ylabel("$\dfrac{1}{\epsilon} |x(p + \epsilon) - x(p)|$")
    plt.xlabel("$\epsilon$")
    plt.legend()
    plt.show()
