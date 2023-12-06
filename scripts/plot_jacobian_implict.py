import sys
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))

from domain_specific.evalf import evalf
from domain_specific.jacobian import evalJacobian
from utils.performance import measure_speed, measure_mem, measure_eps_effect_gcr


def plot_eps_effect():
    triples, b = measure_eps_effect_gcr(np.geomspace(1e-16, 1e+4, 100), n=10)
    #triples, b = measure_eps_effect_gcr(np.geomspace(1e-12, 1e-1, 10), n=10)
    #triples, b = measure_eps_effect_gcr([1e-10, 1e-2], n=10)
    _, axs = plt.subplots(2, sharex=True)
    plt.sca(axs[0])
    plt.plot(triples.eps, triples.iterations.astype(int))
    plt.xscale('log')
    plt.xlabel('eps')
    plt.ylabel('iterations to converge')
    plt.title('Iterations for Implicit Jacobian Solver to Converge\nas a function of Epsilon')
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
    #plt.legend()
    plt.yscale('log')
    return


def plot_mem():
    evalf_kwargs = {'x': None, 't': None, 'p': None, 'u': None}
    evalJacobian_kwargs = {'x': None, 'p': None, 'u': None}
    f_size, J_size, f_peak, J_peak = measure_mem(range(2, 100), evalf, evalf_kwargs, evalJacobian, evalJacobian_kwargs)
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
    return


def plot_speed():
    evalf_kwargs = {'x': None, 't': None, 'p': None, 'u': None}
    evalJacobian_kwargs = {'x': None, 'p': None, 'u': None}
    f_time, J_time = measure_speed(range(2, 100), evalf, evalf_kwargs, evalJacobian, evalJacobian_kwargs)
    plt.plot(f_time, label="Implicit Jacobian")
    plt.plot(J_time, label="Explicit Jacobian")
    plt.legend()
    plt.xlabel("Size of input")
    plt.ylabel("Time to compute Jacobian (s)")
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Speed improvement using implicit Jacobian")
    return


if __name__ == '__main__':
    plot_speed()
    plt.savefig('implicit_jacobian_speed.png', bbox_inches='tight')
    plt.show()
    #plot_eps_effect()
    #plt.savefig('implicit_jacobian_eps.png', bbox_inches='tight')
    #plt.show()
    plot_mem()
    plt.savefig('implicit_jacobian_mem.png', bbox_inches='tight')
    plt.show()