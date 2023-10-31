from jacobian_implicit import tgcr, jf_product
from test_evalf import evalf, generate_inputs
from jacobian import evalJacobian
from sys import getsizeof
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import memray
import os
import pathlib

def test_jacobian_implicit():
    n = 2
    x, p, u = generate_inputs(n)
    r = x.copy()
    r[1] = [0.5, 0.5]
    r = r.reshape(-1, )
    x = x.reshape(-1, )
    J = evalJacobian(x, p, u)
    Jr_explicit = J.dot(r)
    print(Jr_explicit)
    x = x.reshape(-1, )
    Jr_implicit = jf_product(x, p, u, r, eps=1e-10)
    print(Jr_implicit)
    return


def measure_eps_effect():
    n = 10
    triples = []
    for eps in np.geomspace(1e-12, 1e-1, 100):
        x, p, u = generate_inputs(n)
        x = x.reshape(-1, )
        b = np.array([
            np.geomspace(0.5, 1.5, n),  # y, true nodal
            np.geomspace(.75, 1.25, n), # tilde_y, effective currency
            np.ones(n),                 # mu, currency drift
            ]).reshape(-1, )
        x, r_norms = tgcr(jf_product, b, x, p, u, tolrGCR=1e-12, MaxItersGCR=100, eps=eps)
        #plt.plot(r_norms, label=eps, linewidth=10, alpha=0.5)
        last_norm = r_norms[-1]
        if np.isnan(last_norm):
            last_norm = r_norms[-2]
        triples += [[eps, len(r_norms) - 1, last_norm]]
    triples = pd.DataFrame(triples, columns="eps n final".split())
    _, axs = plt.subplots(2, sharex=True)
    plt.sca(axs[0])
    plt.plot(triples.eps, triples.n.astype(int))
    plt.xscale('log')
    plt.xlabel('eps')
    plt.ylabel('iterations to converge')
    test1 = np.sqrt(1e-15) / np.linalg.norm(b)
    test2 = np.sqrt(1e-15 * (1 + np.linalg.norm(b))) / np.linalg.norm(b)
    plt.axvline(x=test1, color="gray", dashes=[1,1], zorder=-10)
    plt.axvline(x=test2, color="gray", dashes=[1,1], zorder=-10)
    plt.sca(axs[1])
    plt.plot(triples.eps, triples.final)
    plt.axvline(x=test1, color="gray", dashes=[1,1], zorder=-10)
    plt.axvline(x=test2, color="gray", dashes=[1,1], zorder=-10)
    plt.xlabel('eps')
    plt.xscale('log')
    plt.ylabel('final residual norm')
    plt.legend()
    plt.yscale('log')
    plt.savefig('implicit_jacobian_eps.png', bbox_inches='tight')
    plt.show()
    return

def memray_eval(f):
    if pathlib.Path('output_file.bin').exists():
        pathlib.Path('output_file.bin').unlink()
    with memray.Tracker("output_file.bin"):
        f()
    stats = memray._memray.compute_statistics(
        os.fspath('output_file.bin'),
        report_progress=True,
        num_largest=999,
    )
    return stats.total_memory_allocated, stats.peak_memory_allocated

def measure_mem():
    t = None
    f_size, J_size, f_peak, J_peak = [], [], [], []
    for i in range(2, 100):
        x, p, u = generate_inputs(i)
        x = x.reshape(-1, )
        # Size of output of f (x2?)
        total, peak = memray_eval(lambda: evalf(x, t, p, u))
        f_size.append(total)
        f_peak.append(peak)
        # Size of Jacobian
        total, peak = memray_eval(lambda: evalJacobian(x, p, u))
        J_size.append(total)
        J_peak.append(peak)
    color = plt.plot(f_size, label="Implicit Jacobian")[0].get_color()
    plt.plot(f_peak, color=color, dashes=[1,1], zorder=20)
    color = plt.plot(J_size, label="Explicit Jacobian")[0].get_color()
    plt.plot(J_peak, color=color, dashes=[1,1], zorder=20)
    plt.legend()
    plt.xlabel("Size of input")
    plt.ylabel("Memory (bytes)")
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Memory improvement using implicit Jacobian")
    plt.savefig('implicit_jacobian_mem.png', bbox_inches='tight')
    return


def measure_speed():
    t = None
    f_time, J_time = [], []
    for i in range(2, 100):
        x, p, u = generate_inputs(i)
        x = x.reshape(-1, )
        # Size of output of f (x2?)
        tic = time.time()
        f = evalf(x, t, p, u)
        toc = time.time()
        f_time.append(toc - tic)
        # Size of Jacobian
        tic = time.time()
        J = evalJacobian(x, p, u)
        toc = time.time()
        J_time.append(toc - tic)
    plt.plot(f_time, label="Implicit Jacobian")
    plt.plot(J_time, label="Explicit Jacobian")
    plt.legend()
    plt.xlabel("Size of input")
    plt.ylabel("Time to compute Jacobian (s)")
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Speed improvement using implicit Jacobian")
    plt.savefig('implicit_jacobian_speed.png', bbox_inches='tight')
    return


if __name__ == '__main__':
    #test_jacobian_implicit()
    #measure_mem()
    #measure_speed()
    measure_eps_effect()
