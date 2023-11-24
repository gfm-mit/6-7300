import numpy as np

from domain_specific.evalf import evalf
from domain_specific.jacobian import evalJacobian


def newton_nd(eval_f, x0, p, u,
              errf=float('inf'), errDeltax=float('inf'),
              MaxIter=float('inf'), eval_Jf=None):
    """
    Uses Newton's Method to solve the scalar nonlinear system f(x) = 0.

    :param eval_f: Function to evaluate f for a given x
    :param x0: Initial guess for Newton iteration
    :param p: Structure containing all parameters needed to evaluate f()
    :param u: Values of inputs
    :param eval_Jf: Function to evaluate the Jacobian of f at x (derivative in 1D)
    :param errf: Absolute equation error: how close f to zero
    :param errDeltax: Absolute output error: how close |x_k - x_{k-1}|
    :param MaxIter: Maximum number of iterations allowed
    :param visualize: If 1, shows intermediate results
    :return: x, converged, errf_k, errDeltax_k, relDeltax_k, iterations, X (list of intermediate solutions)
    """
    k = 1  # Newton iteration index
    X = [x0]  # X stores intermediate solutions
    if not isinstance(x0, np.ndarray):
      X = [np.array([x0])]

    f = eval_f(x0, t=None, p=p, u=u)
    errf_k = abs(f)

    errDeltax_k = float('inf') * np.ones_like(errf_k)

    while k <= MaxIter and ((errf_k > errf).any() or (errDeltax_k > errDeltax).any()):
        Jf = eval_Jf(X[k-1], p, u)

        Deltax = np.linalg.solve(Jf, -f)
        X.append(X[k-1] + Deltax)
        k += 1
        f = eval_f(X[k-1], t=None, p=p, u=u)
        errf_k = abs(f)
        errDeltax_k = abs(Deltax)

    x = X[-1]  # returning only the very last solution
    iterations = k - 1  # returning the number of iterations with ACTUAL computation

    converged = (errf_k <= errf).any() and (errDeltax_k <= errDeltax).any()
    if converged:
        print(f'Newton converged in {iterations} iterations')
    else:
        print('Newton did NOT converge! Maximum Number of Iterations reached')

    return x, converged, errf_k, errDeltax_k, iterations, X


def newton_matlab_wrapper(x0, p, u):
    x3 = np.reshape(x0, [-1]).copy()
    x1, converged, errf_k, errDeltax_k, iterations, X = newton_nd(
        evalf, x3, p, u, errf=1e-4, errDeltax=1e-4, eval_Jf=evalJacobian)
    return x1