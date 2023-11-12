import numpy as np

from domain_specific.evalf import evalf
from domain_specific.jacobian import evalJacobian
from newton.from_julia import newton_nd

def continuation_taylor0(x3, q, p0, u0):
    def continuation(x1, t=None, p=None, u=None):
        fq = evalf(x1, t=None, p=p, u=u)
        return x1 * (1 - q) + fq * q
    return continuation

def continuation_taylor1(x3, q, p0, u0):
    def continuation(x1, t=None, p=None, u=None):
        f0 = evalf(x3, t=None, p=p, u=u)
        fq = evalf(x1, t=None, p=p, u=u)
        return f0 * (1 - q) + fq * q
    return continuation

def continuation_taylor2(x3, q, p0, u0):
    J = evalJacobian(x3, p=p0, u=u0)
    f0 = evalf(x3, t=None, p=p0, u=u0)
    def continuation(x1, t=None, p=None, u=None):
        fq = evalf(x1, t=None, p=p, u=u)
        return (f0 + J @ (x1 - x3)) * (1 - q) + fq * q
    return continuation

def newton_continuation_wrapper(x0, p, u, qs, fqs):
    x3 = np.reshape(x0, [-1])

    x1 = x3
    for q in qs:
        fq = fqs(x3, q, p, u)
        x1, converged, errf_k, err_dx_k, rel_dx_k, iterations, X = newton_nd(
            fq, x1, p, u,
            errf=1e-4, err_dx=1e-4,
            max_iter=1e5,
            #eval_jf=evalJacobian,
            fd_tgcr_params=dict(
                tolrGCR=1e-4,
                MaxItersGCR=1e5,
                eps=1e-4,
            ))
    return x1