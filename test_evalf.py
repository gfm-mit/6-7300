from scipy.integrate import odeint
import numpy as np
from evalf import evalf, generate_inputs
import einops
from plot_util import plot_evolution


def test_symmetric_equilibrium():
    x, p, u = generate_inputs(2)
    x0 = np.array([
        [1, 1], 
        [1, 1],
        [0, 0]
        ]).reshape(-1,)
    dx = evalf(x0, None, p, u)
    dx = einops.rearrange(dx, "(n d) -> d n", d=2)
    assert (dx == 0).all()


def test_convergence():
    T = 1000
    x0 = np.array([
        [1, 1.1],
        [1, 1.1],
        [0, 0]
        ]).reshape(-1,)
    t = np.linspace(0, T, T)
    ans = runode(x0, t)[0][999].reshape(3, 2)
    n1_ans = ans[:, 0]
    n2_ans = ans[:, 1]
    assert(round(n1_ans[0], 13) == round(n2_ans[0], 13))
    assert(round(n1_ans[1], 13) == round(n2_ans[1], 13))


def runode(x0, t, n=3):
    x, p, u = generate_inputs(n)
    ans = odeint(evalf, x0, t, args=(p, u), full_output=True)
    return ans


# test cases, all with two countries:
# 1) all equal
# 2) all equal, but y_tilde starts too high
# 3) start with slightly different currency values, should converge
# 4) test with very small tau2 / tau_mu
# 5) with large alpha, shouldn't ring
# 6) with small alpha, should ring
if __name__ == '__main__':
    T = 100
    x0 = np.array([
        [1, 1.1], 
        [1, 1.1],
        [0, 0]
        ]).reshape(-1,)
    t = np.linspace(0, T, T)
    x, p, u = generate_inputs(2)
    F = evalf(x0, t, p, u)
    F = np.reshape(F, [3, -1]).transpose()
    ans = runode(x0, t)[0]
    ans = einops.rearrange(ans, "t (d n) -> d n t", d=3)
    print(F)
    #print(ans)
    test_convergence()
    #plot_evolution(ans)
