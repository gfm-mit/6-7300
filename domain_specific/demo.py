import numpy as np
import pandas as pd
# only for testing
import sys
import os
import pathlib

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))

import newton.from_julia
import domain_specific.x0


def generate_wobble_inputs(n, t=100, seed=5):
    x0 = np.zeros([3, n])
    p = domain_specific.x0.generate_demo_parameters(n, t=t)
    u = domain_specific.x0.generate_shocks(n)

    p["tau1"] = 7e1
    p["tau2"] = 5e-3
    p["tau3"] = 1e2 # seems to give US wobble with 30 day period

    p['alpha'] = 1e-1 # determines the eventual shift
    p['sigma'] = np.zeros([n])
    p['gamma2'] = .4 * np.ones([n]) # from IMF estimates

    p['d'][1:, 2, 0] = 5e-2
    p['d'][1:, 0, 2] = 5e-2

    p_steady = p.copy()
    p_steady['d'] = p_steady['d'][0, :, :]
    x1 = newton.from_julia.newton_julia_jacobian_free_wrapper(x0, p_steady, u)

    return x1, p, u