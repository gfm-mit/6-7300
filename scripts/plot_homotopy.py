import sys
import os
import pathlib
import time
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))

from domain_specific.x0 import generate_stochastic_inputs
from newton.homotopy import alpha_tau3, preconditioner, none, newton_continuation_wrapper


def test_speed(fq):
    tic = time.time()
    for _ in range(int(1e2)):
        qs = [0, 1]
        if fq == none:
            # get the right overhead for timing
            qs = [1]
        x0, p, u = generate_stochastic_inputs(50)
        x1 = newton_continuation_wrapper(
            x0, p, u,
            qs=qs,
            fqs=fq)
    toc = time.time()
    return toc - tic

if __name__ == "__main__":
    results = {}
    for fq in tqdm([none, alpha_tau3, preconditioner]):
        results[fq.__name__] = test_speed(fq)
    for k, v in results.items():
        print(k, np.round(v, 4))