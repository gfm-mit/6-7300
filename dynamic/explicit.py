import sys
import os
import pathlib

import numpy as np

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))

from domain_specific.evalf import evalf

def forward_euler(x0, p, u, delta_t):
    return x0 + delta_t * evalf(x0, t=None, p=p, u=u)

def rk4(x0, p, u, delta_t):
    k1 = evalf(x0, t=None, p=p, u=u)
    k2 = evalf(x0 + delta_t/2 * k1, t=None, p=p, u=u)
    k3 = evalf(x0 + delta_t/2 * k2, t=None, p=p, u=u)
    k4 = evalf(x0 + delta_t * k3, t=None, p=p, u=u)
    return x0 + delta_t/6 * (k1 + 2*k2 + 2*k3 + k4)

def simulate(x0, p, u, t1, delta_t, f_step=forward_euler, demo=False):
    ts = list(np.arange(0, t1, delta_t)[1:]) + [t1]
    x1 = np.reshape(x0, [-1])
    yield x1
    for t in ts:
        if demo:
            p_demo = p.copy()
            p_demo['d'] = p_demo['d'][int(t / delta_t) - 1, :, :]
            x1 = f_step(x1.copy(), p_demo, u, delta_t)
        else:
            # TODO: remove this copy
            x1 = f_step(x1.copy(), p, u, delta_t)
        yield x1

def simulate_exports(xs, p, u, t1, delta_t, x0="ignored", f_step="ignored", demo="ignored"):
    ts = list(np.arange(0, t1, delta_t)[1:]) + [t1]
    p_demo = p.copy()
    p_demo['d'] = p_demo['d'][0, :, :]
    _, xm = evalf(xs[0], None, p_demo, u, yield_intermediates=True)
    yield xm
    for t, x in zip(ts, xs):
        p_demo = p.copy()
        p_demo['d'] = p_demo['d'][int(t / delta_t) - 1, :, :]
        _, xm = evalf(x, None, p_demo, u, yield_intermediates=True)
        yield xm