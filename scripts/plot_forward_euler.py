import sys
import os
import pathlib
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
from domain_specific.x0 import generate_deterministic_inputs
from dynamic.explicit import simulate

def plot_forward_euler(log_delta_t=-3):
    x0, p, u = generate_deterministic_inputs(3)
    delta_t = np.power(10, log_delta_t)

    xs = list(simulate(x0, p, u, 20, delta_t))
    xs = np.stack(xs)
    error = xs - golden
    error = np.linalg.norm(error, axis=1, ord=np.inf)

if __name__ == '__main__':
    golden = np.load('tests/dynamic_golden_1e-3.npy')[::100]
    # golden = np.load('/Users/sarthak/programming/modellingsim/6-7300/tests/dynamic_golden_1e-3.npy')[::100]
    ts = np.concatenate([np.arange(0, 20, 1e-5), [20]])
    plt.plot(ts, golden)
    for i in range(0, 4):
        plot_forward_euler(-i)