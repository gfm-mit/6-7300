import sys
import os
import pathlib
import time
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))

from domain_specific.evalf import evalf
from domain_specific.x0 import generate_stochastic_inputs
from newton.homotopy import alpha, alpha_tau3, gold, none, newton_continuation_wrapper, permuted_tridiagonal, preconditioner, tau3


def test_speed(fq):
    tic = time.time()
    for _ in range(int(1e1)):
        qs = [0, 1]
        if fq in [none, preconditioner, permuted_tridiagonal, gold]:
            # fix the overhead if we're not _actually_ using the two-stage continuation
            qs = [0]
        x0, p, u = generate_stochastic_inputs(50)
        x1 = newton_continuation_wrapper(
            x0, p, u,
            qs=qs,
            fqs=fq)
        f = evalf(x1, t=None, p=p, u=u)

        error = np.linalg.norm(f) 
        if error > 1e-4:
            return 1e4
    toc = time.time()
    return toc - tic

if __name__ == "__main__":
    results = {}
    for fq in tqdm([permuted_tridiagonal, none, preconditioner, alpha_tau3, alpha, tau3, gold]):
        results[fq.__name__] = test_speed(fq)
    for k, v in results.items():
        print(k, np.round(v, 4))