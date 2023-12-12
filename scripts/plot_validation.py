import similaritymeasures
import numpy as np
import matplotlib.pyplot as plt
import einops
from scipy.optimize import curve_fit


from domain_specific.x0 import generate_demo_inputs
from dynamic import explicit
from domain_specific.evalf import evalf


countries = {0: 'USA', 1: 'EUR', 2: 'IND', 3: 'CHN', 4: 'MEX', 5: 'CAN', 6: 'BRA', 7: 'SGP', 8: 'AUS', 9: 'GHA'}

# For curve_fit
x0, p, u = generate_demo_inputs(10)


def validate_jcurve():
    kwargs = dict(
        x0=x0,
        p=p,
        u=u,
        t1=100,
        delta_t=1e-2,
        f_step=explicit.rk4,
        demo=True
    )
    xs = np.array(list(explicit.simulate(**kwargs)))
    stacked = einops.rearrange(xs, 't (d c) -> c d t', d=3)
    ns = []
    for i in range(1000, len(xs)-1):
        p_demo = p.copy()
        p_demo['d'] = p['d'][i, :, :]
        _, x_ij = evalf(xs[i], None, p_demo, u, yield_intermediates=True)
        exports = np.sum(x_ij, axis=1)
        imports = np.sum(x_ij, axis=0)
        N = exports / imports
        ns.append(np.log(N[2]))
    popt, pcov = curve_fit(bahmani_oskooee, stacked[2, 0, 1000:len(xs)-1], ns)
    tb = jcurve(popt, stacked[2, 0, :])
    plt.plot(ns, label="Ours")
    exp_data = np.array([[t, n] for t, n in enumerate(ns)])
    num_data = np.array([[t, j] for t, j in enumerate(tb)])
    d = similaritymeasures.mae(exp_data, num_data)
    print(d)
    d = similaritymeasures.mse(exp_data, num_data)
    print(d)
    plt.legend()
    plt.ylabel(r"log($\frac{\text{exports}}{\text{imports}}$)")
    plt.xlabel("Time (days)")
    plt.show()
    return


def jcurve(popt, er):
    beta, gamma, lambd, eps = popt
    tb = []
    for t in range(1000, len(er)-1):
        # er is log of currency value
        tb.append((beta * np.log(p['g'][0])) + (gamma * np.log(p['g'][2])) + (lambd * er[t]) + eps)
    plt.plot(tb, label='Bahmani-Oskooee 1985')
    return tb


def bahmani_oskooee(er, beta, gamma, lambd, eps):
    return (beta * np.log(p['g'][0])) + (gamma * np.log(p['g'][2])) + (lambd * er) + eps


if __name__ == "__main__":
    validate_jcurve()

