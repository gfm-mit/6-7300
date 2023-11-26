import numpy as np

from domain_specific.evalf import evalf
from domain_specific.jacobian import evalJacobian
from linear.tgcr_implicit import tgcr_find_root


def newton_nd(eval_f, x0, p, u,
              errf=float('inf'),
              err_dx=float('inf'),
              rel_dx=float('inf'),
              max_iter=float('inf'),
              max_x_norm=float('inf'),
              eval_jf=None,
              fd_tgcr_params=None,
              step_size=None,
              verbose=True):
    k = 1  # Newton method iteration index
    X = [x0]  # Storing intermediate solutions

    f = eval_f(x0, t=None, p=p, u=u)
    errf_k = np.linalg.norm(f, np.inf)

    err_dx_k = np.inf
    rel_dx_k = np.inf

    xk = x0
    while k <= max_iter and (errf_k > errf or err_dx_k > err_dx or rel_dx_k > rel_dx):
        if eval_jf is not None:
            Jf = eval_jf(xk, p=p, u=u)
            dx = np.linalg.solve(Jf, -f)
        else:
            dx, r_norms = tgcr_find_root(x0=x0, p=p, u=u, eval_f=eval_f, **fd_tgcr_params, verbose=verbose)

        if step_size is not None:
            dx *= step_size
        xk += dx
        X += [xk]

        k += 1
        f = eval_f(xk, t=None, p=p, u=u)
        errf_k = np.linalg.norm(f, np.inf)
        err_dx_k = np.linalg.norm(dx, np.inf)
        x_norm = np.linalg.norm(xk, np.inf)
        rel_dx_k = err_dx_k / x_norm
        if x_norm > max_x_norm:
            print("Newton diverged: x value greater than {}".format(max_x_norm))
            break

    x = xk  # Extracting the last solution
    iterations = k - 1

    converged = (errf_k <= errf) and (err_dx_k <= err_dx) and (rel_dx_k <= rel_dx)
    if verbose:
        print("\nNewton converged in {} iterations".format(iterations) if converged else "Newton method reached maximum iterations, {} without convergence".format(iterations))
        print("Norm values in the final iteration are:")
        print("||f(x)||_∞ = {}, given limit is: {}".format(errf_k, errf))
        print("||Δx||_∞ = {}, given limit is: {}".format(err_dx_k, err_dx))
        print("||Δx||_∞/||x||_∞ = {}, given limit is: {}".format(rel_dx_k, rel_dx))

    return x, converged, errf_k, err_dx_k, rel_dx_k, iterations, X

def newton_julia_stepsize_wrapper(x0, p, u):
    x3 = np.reshape(x0, [-1]).copy()

    x1, converged, errf_k, err_dx_k, rel_dx_k, iterations, X = newton_nd(
        evalf, x3, p, u,
        errf=1e-4, err_dx=1e-4,
        max_iter=100,
        step_size=1e-1,
        max_x_norm=1e3,
        eval_jf=evalJacobian)
    return x1

def newton_julia_wrapper(x0, p, u):
    x3 = np.reshape(x0, [-1]).copy()

    x1, converged, errf_k, err_dx_k, rel_dx_k, iterations, X = newton_nd(
        evalf, x3, p, u,
        errf=1e-4, err_dx=1e-4,
        max_iter=10,
        max_x_norm=1e3,
        eval_jf=evalJacobian)
    return x1

def newton_julia_jacobian_free_wrapper(x0, p, u):
    x3 = np.reshape(x0, [-1]).copy()

    x1, converged, errf_k, err_dx_k, rel_dx_k, iterations, X = newton_nd(
        evalf, x3, p, u,
        errf=1e-4, err_dx=1e-4,
        max_iter=10,
        max_x_norm=1e3,
        fd_tgcr_params=dict(
            tolrGCR=1e-4,
            MaxItersGCR=1e5,
            eps=1e-4,
        ))
    return x1