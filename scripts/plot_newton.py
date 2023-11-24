import sys
import os
import pathlib

import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))

from domain_specific.evalf import evalf
from domain_specific.x0 import generate_inputs, generate_lognormal_input
from domain_specific.jacobian import evalJacobian
from utils.performance import memray_eval
from newton.from_julia import newton_julia_jacobian_free_wrapper, newton_julia_wrapper


def plot_mem():
    f_size, J_size, f_time, J_time = [], [], [], []
    t = None
    xs = np.arange(2, 6).astype(int)
    for i in tqdm(xs, desc="mem"):
        x, p, u = generate_lognormal_input(i)
        x = x.reshape(-1, )
        # Size of output of f (x2?)
        tic = time.time()
        total, peak = memray_eval(lambda: newton_julia_wrapper(x, p, u))
        toc = time.time()
        f_size.append(total)
        f_time.append(toc - tic)
        # Size of Jacobian
        tic = time.time()
        total, peak = memray_eval(lambda: newton_julia_jacobian_free_wrapper(x, p, u))
        toc = time.time()
        J_size.append(total)
        J_time.append(toc - tic)
    fig, axs = plt.subplots(1, 2)
    plt.sca(axs[0])
    color = plt.plot(xs, f_size, label="Implicit Jacobian")[0].get_color()
    plt.sca(axs[1])
    plt.plot(xs, f_time, color=color, dashes=[1,1], zorder=20)
    plt.sca(axs[0])
    color = plt.plot(xs, J_size, label="Explicit Jacobian")[0].get_color()
    plt.sca(axs[1])
    plt.plot(xs, J_time, color=color, dashes=[1,1], zorder=20)
    plt.sca(axs[0])
    plt.legend()
    plt.xlabel("# Countries")
    plt.ylabel("Memory (bytes)")
    #plt.xscale('log')
    plt.yscale('log')
    plt.title("Memory improvement using implicit Jacobian")
    plt.sca(axs[1])
    plt.gca().yaxis.set_label_position("right")
    plt.gca().yaxis.tick_right()
    plt.xlabel("# Countries")
    plt.ylabel("Time (seconds)")
    return

def plot_surface():
    x, p, u = generate_inputs(3)
    fig, axs = plt.subplots(3, 3)
    x_flat = np.reshape(x, [-1])
    for i in range(3):
        for j in range(3):
            k = i * 3 + j
            deltas = np.linspace(-2, 2, 100)
            dx = np.zeros([9])
            fs = []
            for delta in deltas:
                dx[k] = delta
                fs += [evalf(x_flat + dx, t=None, p=p, u=u)[0]]
            fs = np.array(fs)

            plt.sca(axs[i, j])
            plt.plot(deltas, fs)
            plt.title("dx[{}, {}]".format(i, j))
    plt.tight_layout()

if __name__ == '__main__':
    #plot_mem()
    #plt.savefig('newton_mem.png', bbox_inches='tight')
    #plt.show()
    x, p, u = generate_inputs(3)
    J = evalJacobian(x, p, u)
    print(J.round(1))
    #plot_surface()
    #plt.show()