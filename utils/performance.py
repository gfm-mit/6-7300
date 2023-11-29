import os
import pathlib
from collections.abc import Iterable
import time

import numpy as np
import pandas as pd
import memray
from tqdm import tqdm

from linear.old_tgcr_implicit import jf_product, gcr_implicit_wrapper
from domain_specific.evalf import evalf
from domain_specific.x0 import generate_deterministic_inputs, generate_stochastic_inputs
from domain_specific.jacobian import evalJacobian


def measure_eps_effect_gcr(epsilons, n=10):
    triples = []
    # for some reason, this produces the nice V shape we expected
    # x0, p, u = generate_inputs(n)
    # whereas this does not
    x0, p, u = generate_stochastic_inputs(n)
    x0 = x0.reshape(-1, )
    #f = evalf(x0, t=None, p=p, u=u)
    #J = evalJacobian(x0, p, u)

    for eps in tqdm(epsilons, desc="eps_effect_gcr"):
        x1, r_norms = gcr_implicit_wrapper(x0=x0, p=p, u=u, tolrGCR=1e-4, eps=eps)

        f = evalf(x0, t=None, p=p, u=u)
        J = evalJacobian(x0, p, u)

        rel_error = np.linalg.norm(f + J @ x1) / np.linalg.norm(f)
        triples += [dict(
            eps=eps,
            iterations=len(r_norms) - 1,
            error=rel_error,
            final=r_norms[-1],
            )]
    triples = pd.DataFrame(triples)
    return triples, -f


def measure_speed(ns):
    assert isinstance(ns, Iterable)
    t = None
    f_time, J_time = [], []
    for i in tqdm(ns, desc="measure_speed"):
        x, p, u = generate_deterministic_inputs(i)
        x = x.reshape(-1, )
        # Size of output of f (x2?)
        tic = time.time()
        f = evalf(x, t, p, u)
        toc = time.time()
        f_time.append(toc - tic)
        # Size of Jacobian
        tic = time.time()
        J = evalJacobian(x, p, u)
        toc = time.time()
        J_time.append(toc - tic)
    return f_time, J_time


def measure_eps_effect_one_step(epsilons, n=10):
    assert isinstance(epsilons, Iterable)
    t = None
    df, error = [], []
    for eps in tqdm(epsilons, desc="eps_effect_one_step"):
        x0, p, u = generate_stochastic_inputs(n)
        x0 = x0.reshape(-1, )
        f = evalf(x0, t, p, u)
        # Compute perturbation
        dx0 = np.random.randn(*x0.shape)
        dx0 = dx0 / np.linalg.norm(dx0) * eps

        # Compute f(x0 + dx0)
        f_perturbed = evalf(x0 + dx0, t, p, u)

        # Compute J(x0) dx0
        Jdx = jf_product(x0, p, u, dx0, eps)
        # Compute relative error
        df.append(np.linalg.norm(f - f_perturbed) / np.linalg.norm(f))
        error.append(np.linalg.norm(f + Jdx - f_perturbed) / np.linalg.norm(f))
    return df, error


def memray_eval(f):
    if pathlib.Path('output_file.bin').exists():
        pathlib.Path('output_file.bin').unlink()
    with memray.Tracker("output_file.bin"):
        f()
    stats = memray._memray.compute_statistics(
        os.fspath('output_file.bin'),
        report_progress=True,
        num_largest=999,
    )
    pathlib.Path('output_file.bin').unlink()
    return stats.total_memory_allocated, stats.peak_memory_allocated


def measure_mem(ns):
    f_size, J_size, f_peak, J_peak = [], [], [], []
    t = None
    for i in tqdm(ns, desc="mem"):
        x, p, u = generate_deterministic_inputs(i)
        x = x.reshape(-1, )
        # Size of output of f (x2?)
        total, peak = memray_eval(lambda: evalf(x, t, p, u))
        f_size.append(total)
        f_peak.append(peak)
        # Size of Jacobian
        total, peak = memray_eval(lambda: evalJacobian(x, p, u))
        J_size.append(total)
        J_peak.append(peak)
    return f_size, J_size, f_peak, J_peak