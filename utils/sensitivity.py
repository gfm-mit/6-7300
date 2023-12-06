from dynamic import explicit, implicit
import numpy as np


def analyze_sensitivity(x0, p, u, p_key, dp, t1=40):
    """
    Compute dy/dp_i for parameter p[p_key]
    """
    kwargs = dict(
        x0=x0,
        p=p,
        u=u,
        t1=t1,
        delta_t=1e-2,
        f_step=explicit.rk4,
    )
    xs = np.array(list(explicit.simulate(**kwargs)))
    # Note some parameters are scalars and others are vectors
    kwargs['p'][p_key] += dp
    xs_perturb = np.array(list(explicit.simulate(**kwargs)))
    # Returns sensitivity over time
    # np.subtract(xs_perturb, xs) / dp
    return xs, xs_perturb

