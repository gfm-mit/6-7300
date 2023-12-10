import einops
import numpy as np
import scipy

from domain_specific.evalf import evalf
from domain_specific.jacobian import evalJacobian
from newton.from_julia import newton_nd


def standard(x3, q, p0, u0):
    f3 = evalf(x3, t=None, p=p0, u=u0)
    def continuation(x1, t=None, p=None, u=None):
        fq = evalf(x1, t=None, p=p, u=u)
        df = x1 - x3
        return (f3 - df) * (1 - q) + fq * q
    return continuation


def taylor(x3, q, p0, u0):
    J = evalJacobian(x3, p=p0, u=u0)
    f3 = evalf(x3, t=None, p=p0, u=u0)
    def continuation(x1, t=None, p=None, u=None):
        fq = evalf(x1, t=None, p=p, u=u)
        df = J @ (x1 - x3)
        return (f3 + df) * (1 - q) + fq * q
    return continuation


def alpha(x3, q, p0, u0):
    def continuation(x1, t=None, p=None, u=None):
        p2 = p0.copy()
        p2['alpha'] = p0['alpha'] * np.power(0.1, 10*(1-q))
        fq = evalf(x1, t=None, p=p2, u=u)
        return fq
    return continuation


def tau1(x3, q, p0, u0):
    def continuation(x1, t=None, p=None, u=None):
        p2 = p0.copy()
        p2['tau1'] = p0['tau1'] * np.power(10, 3*(1-q))
        fq = evalf(x1, t=None, p=p2, u=u)
        return fq
    return continuation


def tau3(x3, q, p0, u0):
    def continuation(x1, t=None, p=None, u=None):
        p2 = p0.copy()
        p2['tau3'] = p0['tau3'] * np.power(0.1, 10*(1-q))
        fq = evalf(x1, t=None, p=p2, u=u)
        return fq
    return continuation


def alpha_tau3(x3, q, p0, u0):
    p3 = p0.copy()
    p3['alpha'] = p0['alpha'] * np.power(0.1, 10*(1-q))
    p3['tau3'] = p0['tau3'] * np.power(0.1, 10*(1-q))
    J = evalJacobian(x3, p=p3, u=u0)
    P = np.linalg.inv(J)
    
    def continuation(x1, t=None, p=None, u=None):
        p2 = p0.copy()
        p2['alpha'] = p0['alpha'] * np.power(0.1, 10*(1-q))
        p2['tau3'] = p0['tau3'] * np.power(0.1, 10*(1-q))
        fq = evalf(x1, t=None, p=p2, u=u)
        return P @ fq
    return continuation


def preconditioner(x3, q, p0, u0):
    J = evalJacobian(x3, p=p0, u=u0)
    lambdas = (J @ x3) / x3
    P = np.diag(1 / lambdas)

    def continuation(x1, t=None, p=None, u=None):
        fq = evalf(x1, t=None, p=p, u=u)
        return (P @ fq) * (1 - q) + fq * q
    return continuation


def permuted_tridiagonal(x3, q, p0, u0):
    J = evalJacobian(x3, p=p0, u=u0)
    n = x3.shape[0] // 3

    P1 = einops.rearrange(J, '(d n) (c m) -> (n d) (m c)', d=3, c=3)

    # turns out this is slightly slower than np.linalg.inv for tridiagonal some reason, but basically the same
    banded = np.stack([
        np.pad(np.diag(P1, k=1), [1, 0]),
        np.diag(P1),
        np.pad(np.diag(P1, k=1), [0, 1]),
    ])
    P2 = scipy.linalg.solve_banded((1, 1), banded, np.eye(3 * n))

    P3 = einops.rearrange(P2, '(n d) (m c) -> (d n) (c m)', d=3, c=3)

    def continuation(x1, t=None, p=None, u=None):
        fq = evalf(x1, t=None, p=p, u=u)
        return (P3 @ fq) * (1 - q) + fq * q
    return continuation


def only_x_xm(x3, q, p0, u0):
    J = evalJacobian(x3, p=p0, u=u0)
    n = J.shape[0] // 3
    JJ = J[2*n:, n:2*n]
    JJ = np.linalg.inv(JJ)

    P = np.diag(1 / np.diag(J))
    P[2*n:, n:2*n] = JJ

    def continuation(x1, t=None, p=None, u=None):
        fq = evalf(x1, t=None, p=p, u=u)
        return (P @ fq) * (1 - q) + fq * q
    return continuation


def no_x_xm(x3, q, p0, u0):
    J = evalJacobian(x3, p=p0, u=u0)
    n = J.shape[0] // 3
    JJ = J.copy()
    JJ[2*n:, n:2*n] = 0
    P = np.linalg.inv(JJ)

    def continuation(x1, t=None, p=None, u=None):
        fq = evalf(x1, t=None, p=p, u=u)
        return (P @ fq) * (1 - q) + fq * q
    return continuation


def gold(x3, q, p0, u0):
    J = evalJacobian(x3, p=p0, u=u0)
    P = np.linalg.inv(J)

    def continuation(x1, t=None, p=None, u=None):
        fq = evalf(x1, t=None, p=p, u=u)
        return (P @ fq) * (1 - q) + fq * q
    return continuation


def mu_only(x3, q, p0, u0):
    n1 = x3.shape[0] // 3
    n2 = 2 * n1
    # never quite zero, to avoid singularity
    qq = np.exp(-10 * (1 - q))
    def continuation(x1, t=None, p=None, u=None):
        x2 = x1.copy()
        x2[:n1] = 0 # set spot prices are zero
        x2[n2:] = 0 # and mu to zero
        fm = evalf(x2, t=None, p=p, u=u)
        fm[:n2] = 0 # keep only the effect on mu

        fq = evalf(x1, t=None, p=p, u=u)
        return fm * (1 - qq) + fq * qq
    return continuation


def none(x3, q, p0, u0):
    return evalf


def newton_continuation_wrapper(x0, p, u, qs, fqs):
    x3 = np.reshape(x0, [-1])

    x1 = x3.copy()
    for q in qs:
        fq = fqs(x3, q, p, u)
        x1, converged, errf_k, err_dx_k, rel_dx_k, iterations, X = newton_nd(
            fq, x1, p, u,
            errf=1e-4,
            err_dx=1e-4,
            max_iter=10,
            fd_tgcr_params=dict(
                tolrGCR=1e-4,
                MaxItersGCR=1e5,
                eps=1e-7,
            ),
            verbose=False)
    assert converged, f"Homotopy: Newton continuation failed to converge with fqs={fqs.__name__} qs={qs}"
    return x1