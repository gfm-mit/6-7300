import sys
import os
import pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
import time
import einops
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
import pandas as pd

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
from domain_specific.evalf import evalf, get_exports
from domain_specific.x0 import generate_deterministic_inputs, generate_stochastic_inputs
from dynamic import explicit, implicit

def stochastic_parameter_value():
    n = 100
    t1 = 40
    delta_t = 1e-2
    x0, p, u = generate_stochastic_inputs(n)
    deltas = dict(
        alpha=np.random.lognormal(mean=-2*2.3, sigma=1),
        tau1=np.random.lognormal(sigma=1),
        tau2=np.random.lognormal(sigma=1),
        tau3=np.random.lognormal(mean=-1*2.3, sigma=1),
        gamma2=np.random.uniform(.4, .5),
    )
    for k in deltas:
        if isinstance(p[k], np.ndarray):
            p[k] = np.ones_like(p[k]) * deltas[k]
        else:
            p[k] = deltas[k]
    kwargs = dict(
        x0=x0,
        p=p,
        u=u,
        t1=t1,
        delta_t=delta_t,
        guess=explicit.rk4,
        evalf_converter=implicit.get_trapezoid_f,
    )
    x3 = np.reshape(x0, [-1])
    #print(np.round(get_exports(x0[1], p), 20))
    #print(einops.rearrange(
    #    np.round(evalf(x3, None, p, u), 20),
    #    '(n d) -> n d', d=3, n=n))
    try:
        #xs = list(tqdm(explicit.simulate(**kwargs)))
        xs = list(implicit.simulate(**kwargs))
        xs = np.stack(xs)
        norm = np.linalg.norm(xs[:, n:2*n], np.inf, axis=1)
        norm = norm[-1]
        p["norm"] = norm
        p['dim'] = np.round(xs[-1, n:2*n], 2)
        return p
    except AssertionError as e:
        p["norm"] = 1e4
        return p

def test_random_parameters():
    res = []
    for _ in tqdm(range(3)):
        res += [stochastic_parameter_value()]
    res = pd.DataFrame(res)
    res.gamma2 = res.gamma2.map(lambda x: np.median(x))
    return res

def plot_results(res):
    fig, axs = plt.subplots(2, 3)
    for k, ax in zip("alpha gamma2 tau1 tau2 tau3".split(), axs.flatten()):
        plt.sca(ax)
        v = res[k]
        plt.scatter(v, res.norm, alpha=0.2)
        plt.xlabel(k)
        plt.yscale('log')
        if "tau" in k or "alpha" in k:
            plt.xscale('log')
    plt.title("\n".join(map(str, [np.min(res.norm), np.max(res.norm)])))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    res = test_random_parameters()
    res.to_csv('divergence_run.csv')
    res = pd.read_csv('divergence_run.csv')
    plot_results(res)