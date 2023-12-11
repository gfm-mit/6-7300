from domain_specific.x0 import generate_stochastic_inputs, generate_demo_inputs
import matplotlib.pyplot as plt
import einops
import numpy as np
from utils import sensitivity
import seaborn as sns
from tqdm import tqdm


countries = {0: 'USA', 1: 'EUR', 2: 'IND', 3: 'CHN', 4: 'MEX', 5: 'CAN', 6: 'BRA', 7: 'SGP', 8: 'AUS', 9: 'GHA'}

parameters = {'tau1': r'$\tau_1$', 'tau2': r'$\tau_2$', 'tau3': r'$\tau_3$', 'alpha': r'$\alpha$',
             'gamma2': r'$\gamma_2$', 'd': '$d$', 'g': '$g$'}


def visualize_integration(p_key, savefig=None, n=3):
    #eps = [1e-1, 7e-2, 5e-2, 3e-2, 1e-2, 7e-3, 5e-3, 3e-3, 1e-4]
    eps = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 1e-4, 1e-6, 1e-8]
    x0, p, u = generate_demo_inputs(n)
    xs, _ = sensitivity.analyze_sensitivity(x0, p, u, p_key, 0, t1=t1)
    stacked = einops.rearrange(xs, 't (d c) -> c d t', d=3)
    colors = []
    for i in range(n):
        c = plt.plot(np.exp(stacked[i, 0]), label=f"%s" % countries[i])[0].get_color()
        colors.append(c)

    for dp in tqdm(eps):
        xs, xs_perturb = sensitivity.analyze_sensitivity(x0, p, u, p_key, dp, t1=t1)
        # Visualize trajectory of original and perturbed system
        stacked_perturb = einops.rearrange(xs_perturb, 't (d c) -> c d t', d=3)
        # Iterate over nodes (countries)
        for i in range(n):
            # Plot perturbed currency values
            plt.plot(np.exp(stacked_perturb[i, 0]), color=colors[i], zorder=-10, alpha=0.1)
            print(dp)
            print(countries[i])
            print(np.average((np.exp(stacked_perturb[i, 0]) - np.exp(stacked[i, 0])) / np.exp(stacked[i, 0])))
            print('---')
    plt.legend()
    plt.xlabel('Time (days)')
    plt.ylabel('$y$')
    plt.ylim((0.9, 1.1))
    plt.title(r'Trajectory after Perturbing %s' % parameters[p_key])
    plt.legend()
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    plt.clf()


def visualize_perturbation(n, t1, savefig=None, samples=10):
    eps = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 1e-4, 1e-6, 1e-8]
    # Iterate over parameters
    for p_key in parameters.keys():
        data = {'eps': [], 'diff': []}
        avgs = []
        # Iterate over different magnitudes of perturbation
        for dp in tqdm(eps):
            print(dp)
            diffs = []
            # For each perturbation, sample several points to get spread
            for i in range(samples):
                diff = run_perturbation(n, dp, p_key, t1, seed=i)
                diffs.append(diff)
                data['diff'].append(diff)
                data['eps'].append(dp)
            avgs.append(np.median(diffs))
        # Plot average effect
        plt.plot(eps, avgs, label="%s" % parameters[p_key])
        print(p_key)
        print(np.average(avgs))

        # Plot spread of effects
        plt.xscale("log")
        p1 = sns.scatterplot(data=data, x='eps', y='diff',
                            alpha=.2, legend=True, label="%s" % parameters[p_key]
                            )
        p1.set_xscale("log")
    plt.ylabel("$\dfrac{1}{\epsilon} |x(p + \epsilon) - x(p)|$")
    plt.xlabel("$\epsilon$")
    plt.title("Parameter Sensitivity Analysis")
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    return


def run_perturbation(n, dp, p_key, t1, seed=5):
    x0, p, u = generate_demo_inputs(n, seed=seed)
    xs, xs_perturb = sensitivity.analyze_sensitivity(x0, p, u, p_key, dp, t1=t1)
    diff = np.abs(np.subtract(xs_perturb, xs) / dp) # t x 3n
    # Iterate over countries
    country_diffs = []
    for i in range(n):
        country_diffs.append(np.max(diff[:, i])) # Max of y over country i over time
    max_diff = np.max(country_diffs) # Max of y over all countries over time
    return max_diff


if __name__ == "__main__":
    n = 10
    t1 = 100
    #visualize_perturbation(n, t1, savefig="perturbation.png")
    visualize_integration('d', n=4, savefig="integration.png")
