import numpy as np
import matplotlib.pyplot as plt
import einops
from scipy.optimize import curve_fit
from scipy.stats import sem
from sklearn.metrics import mean_squared_error


from domain_specific.x0 import generate_demo_inputs
from dynamic import explicit
from domain_specific.evalf import evalf
import warnings


countries = {0: 'USA', 1: 'EUR', 2: 'IND', 3: 'CHN', 4: 'MEX', 5: 'CAN', 6: 'BRA', 7: 'SGP', 8: 'AUS', 9: 'GHA'}

# For curve_fit
# Bad but sue me
_, p_global, _ = generate_demo_inputs(10)


def run_validation(n=10):
    fig, axs = plt.subplots(2, 4)
    x0, p, u = generate_demo_inputs(n)
    errs = []
    for i, a in enumerate([2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1]):
        p['alpha'] = a
        tb, ns, mse = validate_jcurve(x0, p, u)
        axs[i // 4, i % 4].plot(ns[3000:6000], label="Ours" if i == 3 else "", color="blue", alpha=0.3)
        axs[i // 4, i % 4].plot(tb[3000:6000], label='Bahmani-Oskooee 1985' if i == 3 else "", color="green", alpha=0.5)
        axs[i // 4, i % 4].set_title(r'$\alpha=%.1f$, MSE %.5f' % (a, mse))
        axs[i // 4, i % 4].set_ylim((-0.4, 0.2))
        if i == 0 or i == 4:
            axs[i // 4, i % 4].set_ylabel(r"$\text{log}\frac{\text{exports}}{\text{imports}}$")
        if i == 3:
            axs[i // 4, i % 4].legend(loc="upper right")
        if i >= 4:
            axs[i // 4, i % 4].set_xlabel("Time (days)")
        errs.append(mse)
    fig.suptitle(r'Fit of Impulse Response, MSE %.5f $\pm$ %.5f' % (np.mean(errs), sem(errs)))
    plt.show()
    return


def validate_jcurve(x0, p, u):
    # For USA/IND impulse response
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, pcov = curve_fit(bahmani_oskooee, stacked[2, 0, 1000:len(xs)-1], ns)
    tb = jcurve(popt, stacked[2, 0, :])
    mse = mean_squared_error(ns, tb)
    print(f"MSE: {mse}%")
    return tb, ns, mse


def jcurve(popt, er):
    beta, gamma, lambd, eps = popt
    tb = []
    for t in range(1000, len(er)-1):
        # er is log of currency value
        tb.append((beta * np.log(p_global['g'][0])) + (gamma * np.log(p_global['g'][2])) + (lambd * er[t]) + eps)
    return tb


def bahmani_oskooee(er, beta, gamma, lambd, eps):
    return (beta * np.log(p_global['g'][0])) + (gamma * np.log(p_global['g'][2])) + (lambd * er) + eps


if __name__ == "__main__":
    run_validation()

