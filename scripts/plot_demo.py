import numpy as np
import matplotlib.pyplot as plt
import einops

from domain_specific.x0 import generate_stochastic_real_inputs, generate_demo_inputs, generate_stochastic_inputs, generate_demo_inputs
from domain_specific.evalf import evalf
from dynamic import explicit, implicit


countries = {0: 'USA', 1: 'EUR', 2: 'IND', 3: 'CHN', 4: 'MEX', 5: 'CAN', 6: 'BRA', 7: 'SGP', 8: 'AUS', 9: 'GHA'}


def visualize_real(n, t1=100):
    x0, p, u = generate_demo_inputs(n)
    kwargs = dict(
        x0=x0,
        p=p,
        u=u,
        t1=t1,
        delta_t=1e-2,
        f_step=explicit.rk4,
        demo=True
    )
    xs = np.array(list(explicit.simulate(**kwargs)))
    stacked = einops.rearrange(xs, 't (d c) -> c d t', d=3)
    for i in range(n):
        plt.plot(np.exp(stacked[i, 0]), label=f"%s" % countries[i])
    plt.title("Shock between USA and EUR")
    plt.ylabel("$y$")
    plt.xlabel("Time")
    plt.axvline(x=5000, linestyle='--', color='grey')
    plt.ylim((0.9, 1.1))
    plt.legend()
    plt.show()
    return


def visualize_real_exports(n, t1=100):
    x0, p, u = generate_demo_inputs(n)
    kwargs = dict(
        x0=x0,
        p=p,
        u=u,
        t1=t1,
        delta_t=1e-2,
        f_step=explicit.rk4,
        demo=True
    )
    xs = np.array(list(explicit.simulate(**kwargs)))
    for c in range(n):
        ns = []
        for i in range(len(xs)-1):
            p_demo = p.copy()
            p_demo['d'] = p['d'][i, :, :]
            _, x_ij = evalf(xs[i], None, p_demo, u, yield_intermediates=True)
            exports = np.sum(x_ij, axis=1)
            imports = np.sum(x_ij, axis=0)
            N = exports - imports
            ns.append(N[c])
        plt.plot(ns, label=f"%s" % countries[c])
    plt.legend()
    plt.ylabel("Net exports")
    plt.xlabel("Time")
    plt.show()
    return


if __name__ == '__main__':
    n = 10
    #visualize_real(n)
    visualize_real_exports(n)
