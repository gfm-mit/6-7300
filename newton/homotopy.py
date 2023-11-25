import numpy as np

from domain_specific.evalf import evalf
from domain_specific.jacobian import evalJacobian
from newton.from_julia import newton_nd


def standard(x3, q, p0, u0):
    f3 = evalf(x3, t=None, p=p0, u=u0)
    def continuation(x1, t=None, p=None, u=None):
        fq = evalf(x1, t=None, p=p, u=u)
        df = x1 - x3
        return (f3 + df) * (1 - q) + fq * q
    return continuation


def diag(x3, q, p0, u0):
    f3 = evalf(x3, t=None, p=p0, u=u0)
    eps = 1e-4
    dx = eps * np.ones([9])
    f3_eps = evalf(x3 + dx, t=None, p=p0, u=u0)
    df3 = (f3_eps - f3) / eps

    def continuation(x1, t=None, p=None, u=None):
        fq = evalf(x1, t=None, p=p, u=u)
        df = (x1 - x3) * df3
        return (f3 + df) * (1 - q) + fq * q
    return continuation


def taylor(x3, q, p0, u0):
    J = evalJacobian(x3, p=p0, u=u0)
    f3 = evalf(x3, t=None, p=p0, u=u0)
    def continuation(x1, t=None, p=None, u=None):
        fq = evalf(x1, t=None, p=p, u=u)
        df = J @ (x1 - x3)
        return (f3 + df) * (1 - q) + fq * q
    return continuation


def newton_continuation_wrapper(x0, p, u, qs, fqs):
    x3 = np.reshape(x0, [-1])

    x1 = x3.copy()
    for q in qs:
        fq = fqs(x3, q, p, u)
        x1, converged, errf_k, err_dx_k, rel_dx_k, iterations, X = newton_nd(
            fq, x1, p, u,
            errf=1e-4,
            err_dx=1e-4,
            max_iter=2,
            fd_tgcr_params=dict(
                tolrGCR=1e-4,
                MaxItersGCR=1e5,
                eps=1e-4,
            ))
        assert converged, "Newton continuation failed to converge"
    return x1