import sys
import os
import pathlib
import time

import numpy as np
import pytest


sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))

from domain_specific.evalf import evalf
from domain_specific.x0 import generate_stochastic_inputs
from newton.homotopy import alpha, alpha_tau3, mu_only, preconditioner, none, standard, tau1, tau3, taylor, newton_continuation_wrapper


def pytest_generate_tests(metafunc):
    metafunc.parametrize("fq", [none, preconditioner, alpha_tau3, alpha, tau3, tau1, standard, taylor, mu_only])


def test_homotopy(fq):
    for _ in range(10):
        x0, p, u = generate_stochastic_inputs(3)

        x1 = newton_continuation_wrapper(
            x0, p, u,
            qs=[0, 1],
            fqs=fq)
        f = evalf(x1, t=None, p=p, u=u)

        error = np.linalg.norm(f) 
        assert error < 1e-7, error