import time
import seaborn as sns
from matplotlib import pyplot as plt
import sys
import os
import pathlib
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))

from domain_specific.evalf import evalf
from newton.from_julia import newton_nd
import dynamic.explicit as explicit

def implicit_step(eval_f, x0, p, u, eval_Jf=None):
    x, converged, errf_k, err_dx_k, rel_dx_k, iterations, X = newton_nd(
        eval_f, x0, p, u,
        # WARNING! untuned magical parameters
        errf=1e-4,
        err_dx=1e-4,
        rel_dx=float('inf'),
        max_iter=10,
        eval_jf=eval_Jf,
        fd_tgcr_params=dict(
            tolrGCR=1e-1,
            MaxItersGCR=100,
            eps=1e-10,
        ),
        verbose=False)
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

def simulate(x0, p, u, t1, delta_t, evalf_converter=get_trapezoid_f, guess=explicit.forward_euler):
    ts = list(np.arange(0, t1, delta_t)[1:]) + [t1]
    x1 = np.reshape(x0, [-1])
    yield x1
    for t in tqdm(ts):
        # TODO: remove this copy
        f_step = evalf_converter(x1.copy(), delta_t)
        x1 = guess(x1.copy(), p, u, delta_t)
        x1 = implicit_step(f_step, x1.copy(), p, u)
        yield x1

def dynamic_step(x3, p, u, t1, delta_t, dx_error_max, factory=get_backward_f, guess=explicit.forward_euler):
    clean_x = np.reshape(x3, [-1])
    yield clean_x
    # TODO: repeatedly adding floats is a bad idea
    t = 0
    last_steps = 1
    while t <= t1:
        x0 = clean_x
        f0 = evalf(x0, None, p, u)
        delta_f = 0
        trial_steps = last_steps
        trial_x = guess(x0, p, u, delta_t * trial_steps)
        trial_f = evalf(trial_x, None, p, u)
        delta_f = np.linalg.norm(trial_f - f0)
        if delta_f * trial_steps * delta_t < dx_error_max:
            while delta_f * trial_steps * delta_t < dx_error_max:
                safe_x = trial_x
                safe_steps = trial_steps
                trial_steps = safe_steps * 2
                trial_x = guess(x0, p, u, delta_t * trial_steps)
                trial_f = evalf(trial_x, None, p, u)
                delta_f = np.linalg.norm(trial_f - f0)
        # TODO: this might be dangerous for nonmonotonic functions
        else: # step down
            while trial_steps > 1 and delta_f * trial_steps * delta_t > dx_error_max:
                trial_steps = trial_steps // 2
                trial_x = guess(x0, p, u, delta_t * trial_steps)
                trial_f = evalf(trial_x, None, p, u)
                delta_f = np.linalg.norm(trial_f - f0)
            safe_steps = trial_steps
            safe_x = trial_x
        # TODO: clean up handling of the case safe_steps == 1
        f_step = factory(x0, delta_t * safe_steps)
        clean_x = implicit_step(f_step, safe_x, p, u)
        for s in range(1, safe_steps + 1):
            interpolated_x = x0 + (clean_x - x0) * s / safe_steps
            t += delta_t
            yield interpolated_x
            if t > t1:
                break
        last_steps = safe_steps