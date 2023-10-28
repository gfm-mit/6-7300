import numpy as np
from evalf import evalf
from test_evalf import generate_inputs


def tgcr(f, b, x0, p_in, u, tolrGCR, MaxItersGCR, eps=1e-5):
    """
    Generalized conjugate residual method for solving Ax = b
    INPUTS
    f          : function that produces the matrix-vector product Ar
    b          : right hand side
    tolrGCR    : convergence tolerance, terminate on norm(b - Ax) / norm(b) < tolrGCR
    MaxItersGCR: maximum number of iterations before giving up
    OUTPUTS
    x          : computed solution, returns null if no convergence
    r_norms    : vector containing ||r_k||/||r_0|| for each iteration k

    EXAMPLE:
    [x, r_norms] = tgcr(A,b,tol,maxiters)
    """

    # Generate the initial guess for x at iteration k=0
    x = np.zeros_like(b)

    # Set the initial residual to b - Ax^0 = b
    r = b.copy()
    r_norms = [np.linalg.norm(r, 2)]

    k = 0
    p_full = []
    Ap_full = []
    while (r_norms[k] / r_norms[0] > tolrGCR) and (k < MaxItersGCR):
        k += 1
        # Use the residual as the first guess for the ne search direction and compute its image
        p = r.copy()
        Ap = f(evalf, x0, p_in, u, p, eps=eps)

        # Make the new Ap vector orthogonal to the previous Ap vectors,
        # and the p vectors A^TA orthogonal to the previous p vectors.
        # Notice that if you know A is symmetric
        # you can save computation by limiting the for loop to just j=k-1
        # however if you need relative accuracy better than  1e-10
        # it might be safer to keep full orthogonalization even for symmetric A
        if k > 1:
            for j in range(k - 1):
                beta = np.dot(Ap, Ap_full[j])
                p -= beta * p_full[j]
                Ap -= beta * Ap_full[j]

        # Make the orthogonal Ap vector of unit length, and scale the
        # p vector so that A * p  is of unit length
        # if issparse(Ap):
        #     norm_Ap = scipy.sparse.linalg(Ap, 2)
        # else:
        #     norm_Ap = np.linalg.norm(Ap, 2)
        norm_Ap = np.linalg.norm(Ap, 2)
        Ap = Ap/norm_Ap
        p = p/norm_Ap

        p_full.append(p)
        Ap_full.append(Ap)

        # Determine the optimal amount to change x in the p direction by projecting r onto Ap
        alpha = np.dot(r, Ap)

        # Update x and r
        x = x + alpha * p
        r = r - alpha * Ap

        # Save the norm of r
        r_norms.append(np.linalg.norm(r, 2))

    if r_norms[k] > tolrGCR * r_norms[0]:
        print('GCR did NOT converge! Maximum Number of Iterations reached')
        x = None==8
        print(f'GCR converged in {k} iterations')

    r_norms = np.array(r_norms) / r_norms[0]
    return x, r_norms


def jf_product(f, x0, p, u, r, eps=1e-5):
    t = None
    f1 = f(x0 + (eps * r), t, p, u)
    f0 = f(x0, t, p, u)
    Jr = (1 / eps) * (f1 - f0)
    return Jr


if __name__ == '__main__':
    n = 2
    x, p, u = generate_inputs(n)
    # x = [y, tilde_y, mu]
    # solving for tilde_y
    r = x.copy()
    r[1] = [0.5, 0.5]
    r = r.reshape(-1,)
    x = x.reshape(-1,)
    prod = jf_product(evalf, x, p, u, r)
    b = np.array([[0.5, 1.5],               # y, true nodal
                 [0.75, 1.25],              # tilde_y, effective currency
                 [1, 1]]).reshape(-1, )     # mu, currency drift
    x, r_norms = tgcr(jf_product, b, x, p, u, 10e-2, 100)
    print(x)
