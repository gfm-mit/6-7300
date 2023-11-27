import sys
import os
import pathlib

import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))


from dynamic import explicit, implicit
from domain_specific.evalf import evalf
from domain_specific.x0 import generate_lognormal_input
from domain_specific.jacobian import evalJacobian
from utils import simulation_vis


def plot_simulation(x0, p, u):
    delta_t = 1e-2
    t1 = 100
    kwargs = dict(
        x3=x0,
        p=p,
        u=u,
        t1=t1,
        delta_t=delta_t,
        factory=implicit.get_trapezoid_f,
        guess=explicit.rk4,
        dx_error_max=1e-4
    )
    xs = list(tqdm(implicit.dynamic_step(**kwargs), total=int(t1 / delta_t)))
    simulation_vis.visualize(xs, p, u)


if __name__ == '__main__':
    x0, p, u = generate_lognormal_input(3)
    plot_simulation(x0, p, u)