from domain_specific.x0 import generate_stochastic_inputs
import matplotlib.pyplot as plt
import einops
import numpy as np
from utils import sensitivity
import seaborn as sns
from tqdm import tqdm


def visualize_integration(xs, xs_perturb, savefig=None, n=3):
    # Visualize trajectory of original and perturbed system
    stacked = einops.rearrange(xs, 't (d c) -> c d t', d=3)
    stacked_perturb = einops.rearrange(xs_perturb, 't (d c) -> c d t', d=3)
    # Iterate over nodes (countries)
    for i in range(n):
        # Plot original currency values
        color = plt.plot(stacked[i, 0], label=f"Country {i}")[0].get_color()
        # Plot perturbed currency values
        plt.plot(stacked_perturb[i, 0], color=color, alpha=0.25, linewidth=3, zorder=-10)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Currency value')
    plt.yscale('symlog')
    plt.title('Solid is $y$, dashed lines are $y_p$')
    plt.legend()
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    plt.clf()


def visualize_perturbation(n, t1, p_key, samples=5):
    data = {'eps': [], 'diff': []}
    eps = [10e-8, 10e-10, 10e-12, 10e-14, 10e-16]
    avgs = []
    # Iterate over different magnitudes of perturbation
    for dp in tqdm(eps):
        print(dp)
        diffs = []
        # For each perturbation, sample several points to get spread
        for i in range(samples):
            diff = run_perturbation(n, dp, p_key, t1)
            diffs.append(diff)
            data['diff'].append(diff)
            data['eps'].append(dp)
        avgs.append(np.median(diffs))
    # Plot average effect
    plt.plot(eps, avgs)

    # Plot spread of effects
    plt.xscale("log")
    p1 = sns.scatterplot(data=data, x='eps', y='diff',
                         alpha=.2, legend=True,
                         )
    p1.set_xscale("log")
    plt.ylabel("$\dfrac{1}{\epsilon} |x(p + \epsilon) - x(p)|$")
    plt.xlabel("$\epsilon$")
    plt.title("Perturbing %s" % p_key)
    plt.show()
    return


def run_perturbation(n, dp, p_key, t1):
    x0, p, u = generate_stochastic_inputs(n)
    xs, xs_perturb = sensitivity.analyze_sensitivity(x0, p, u, p_key, dp, t1=t1)
    diff = np.abs(np.subtract(xs_perturb, xs) / dp) # t x 3n
    # Iterate over countries
    country_diffs = []
    for i in range(n):
        country_diffs.append(np.max(diff[:, i])) # Max of y over country i over time
    max_diff = np.max(country_diffs) # Max of y over all countries over time
    # Uncomment to viusalize trajectory
    # visualize(xs, xs_perturb)
    return max_diff


if __name__ == "__main__":
    p_key = 'tau1'
    n = 3
    t1 = 40
    visualize_perturbation(n, t1, p_key)
