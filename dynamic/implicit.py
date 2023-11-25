import seaborn as sns
from matplotlib import pyplot as plt
import sys
import os
import pathlib
import numpy as np

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))

from domain_specific.evalf import evalf
from newton.from_julia import newton_nd
import dynamic.explicit as explicit

def implicit_step(eval_f, x0, p, u, eval_Jf=None):
    print(eval_f(x0, None, p, u))
    x, converged, errf_k, err_dx_k, rel_dx_k, iterations, X = newton_nd(
        eval_f, x0, p, u,
        # WARNING! untuned magical parameters
        errf=1e-4,
        err_dx=1e-4,
        rel_dx=float('inf'),
        max_iter=10,
        eval_jf=eval_Jf,
        fd_tgcr_params=dict(
            tolrGCR=1e-8,
            MaxItersGCR=100,
            eps=1e-10,
        ),
        verbose=False)
    print(eval_f(x, None, p, u))
    return x

def get_backward_f(x0, delta_t):
    def f(x1, t, p, u):
        return x1 - x0 - delta_t * evalf(x1, t, p, u)
    return f

def get_trapezoid_f(x0, delta_t):
    def f(x1, t, p, u):
        f0 = evalf(x0, t, p, u)
        f1 = evalf(x1, t, p, u)
        return x1 - x0 - delta_t / 2 * (f1 + f0)
    return f

def simulate(x0, p, u, t1, delta_t, factory=get_backward_f, guess=explicit.forward_euler):
    ts = list(np.arange(0, t1, delta_t)[1:]) + [t1]
    x1 = np.reshape(x0, [-1])
    yield x1
    for t in ts:
        # TODO: remove this copy
        f_step = factory(x1.copy(), delta_t)
        x1 = guess(x1.copy(), p, u, delta_t)
        x1 = implicit_step(f_step, x1.copy(), p, u)
        yield x1