import numpy as np
import pandas as pd

from domain_specific.evalf import evalf
from domain_specific.x0 import generate_deterministic_inputs, generate_stochastic_inputs
from domain_specific.jacobian import finiteDifferenceJacobian
from linear.gcr import gcrSolver
from linear.bound import getIterationBoundCond, getIterationBoundGirshgorin, getIterationBoundEigen


def runSample(x0, p, u):
    J = finiteDifferenceJacobian(evalf, x0, p, u, delta=1e-6)

    bound1 = getIterationBoundCond(J)
    bound2 = getIterationBoundGirshgorin(J)
    bound3 = getIterationBoundEigen(J)

    x0 = np.reshape(x0, [-1])
    f = evalf(x0, t=None, p=p, u=u)
    x0 = np.reshape(x0, [3, -1])

    x1, _, k = gcrSolver(J, -f)
    rel_error = np.linalg.norm(f + J @ x1) / np.linalg.norm(f)

    res = {'Cond_Bound': bound1, 'Girsh_Bound': bound2, 'Eigen_Bound': bound3, 'Actual #Iterations': k, "rel_error": rel_error}
    return res

def test_samples():
    res = []
    for i in range(30, 31):
        x0, p, u = generate_stochastic_inputs(i)
        row = runSample(x0=x0, p=p, u=u)
        assert row['rel_error'] < 1e-4, row['rel_error']
        row['name'] = "N={}".format(i)
        res.append(row)
    res = pd.DataFrame(res)