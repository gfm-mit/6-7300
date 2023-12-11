import sys
import os
import pathlib

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections.abc import Iterable
import memray

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))

from newton.from_julia import newton_julia_jacobian_free_wrapper, newton_julia_wrapper
from domain_specific.x0 import generate_stochastic_inputs


def measure_speed(ns):
    assert isinstance(ns, Iterable)
    t = None
    matrix_free_time, matrix_time = [], []
    for i in tqdm(ns, desc="measure_speed"):
        for _ in range(1 + 100 // i):
            x, p, u = generate_stochastic_inputs(i)
            tic = time.time()
            newton_julia_jacobian_free_wrapper(x, p, u)
            toc = time.time()
            matrix_free_time.append([i, toc - tic])

            # Size of Jacobian
            tic = time.time()
            newton_julia_wrapper(x, p, u)
            toc = time.time()
            matrix_time.append([i, toc - tic])
    return matrix_free_time, matrix_time


def plot_speed():
    mf, m = measure_speed(np.geomspace(2, 200, 10).astype(int))
    mf = pd.DataFrame(mf, columns=["n", "s"]).groupby("n").median()
    m = pd.DataFrame(m, columns=["n", "s"]).groupby("n").median()
    plt.plot(mf.index, mf.s, label="Implicit Jacobian")
    plt.plot(m.index, m.s, label="Explicit Jacobian")
    plt.legend()
    plt.xlabel("Number of Countries")
    plt.ylabel("Time to Compute")
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Speed Comparison at 1e-5 Error Tolerance")
    return


if __name__ == '__main__':
    #plot_memory()
    #plt.savefig('matrix_free_memory.png', bbox_inches='tight')
    #plt.reset()
    plot_speed()
    plt.savefig('matrix_free_speed.png', bbox_inches='tight')
    #plt.show()