from scipy.integrate import odeint
import numpy as np
from evalf import evalf, evalg
import matplotlib.pyplot as plt
import einops
from plot_util import plot_evolution
import sdeint


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
    x, p, u = generate_inputs(2)
    x0 = np.array([
        [1, 1.1],
        [1, 1.1],
        [0, 0]
        ]).reshape(-1,)
    t = np.linspace(0, T, T)
    def f_wrapper(x, t):
        return evalf(x, t, p, u)
    def g_wrapper(x, t):
        g = evalg(x, t, p, u)[:]
        return g
    ans = sdeint.itoint(f_wrapper, g_wrapper, x0, t)[999].reshape(3, 2)

    n1_ans = ans[:, 0]
    n2_ans = ans[:, 1]
    assert(round(n1_ans[0], 2) == round(n2_ans[0], 2))    # Justify rounding with condition number
    assert(round(n1_ans[1], 2) == round(n2_ans[1], 2))    # Noise at 3 decimal points (sigma)


def test_delays():
    T = 100
    # Small time delay should converge more quickly
    x, p, u = generate_inputs(2)
    p['tau1'] = 1 * np.ones([2])
    x0 = np.array([
        [1, 1.1],
        [1, 1.1],
        [0, 0]
    ]).reshape(-1, )
    t = np.linspace(0, T, T)
    def f_wrapper(x, t):
        return evalf(x, t, p, u)
    def g_wrapper(x, t):
        g = evalg(x, t, p, u)[:]
        return g
    ans_sm = sdeint.itoint(f_wrapper, g_wrapper, x0, t)
    ans_sm = einops.rearrange(ans_sm, "t (d n) -> d n t", d=3)
    avg_sm_oscillation = measure_oscillations(ans_sm)

    # Large time delay should converge more slowly
    x, p, u = generate_inputs(2)
    p['tau1'] = 10 * np.ones([2])
    x0 = np.array([
        [1, 1.1],
        [1, 1.1],
        [0, 0]
    ]).reshape(-1, )
    t = np.linspace(0, T, T)
    ans_lg = sdeint.itoint(f_wrapper, g_wrapper, x0, t)
    ans_lg = einops.rearrange(ans_lg, "t (d n) -> d n t", d=3)
    avg_lg_oscillation = measure_oscillations(ans_lg)
    assert(np.mean(avg_sm_oscillation) < np.mean(avg_lg_oscillation))


def test_elasticity():
    T = 100
    x, p, u = generate_inputs(2)

    # Low elasticity means less oscillations
    p['alpha'] = -1e-1*np.ones([2])
    x0 = np.array([
        [1, 1.1],
        [1, 1.1],
        [0, 0]
        ]).reshape(-1,)
    t = np.linspace(0, T, T)
    ans = runode(x0, t, p, u)[0]
    ans_low = einops.rearrange(ans, "t (d n) -> d n t", d=3)
    avg_low_oscillation = measure_oscillations(ans_low)

    # High elasticity means more oscillations
    p['alpha'] = -2 * np.ones([2])
    x0 = np.array([
        [1, 1.1],
        [1, 1.1],
        [0, 0]
    ]).reshape(-1, )
    t = np.linspace(0, T, T)
    ans = runode(x0, t, p, u)[0]
    ans_high = einops.rearrange(ans, "t (d n) -> d n t", d=3)
    avg_high_oscillation = measure_oscillations(ans_high)
    assert(np.mean(avg_low_oscillation) < np.mean(avg_high_oscillation))


def measure_oscillations(ans):
    oscillation = []
    for i in range(1, 10):
        t_x = ans[:, :, 80 + i]
        t_xplus10 = ans[:, :, (80 + i) + 10]
        oscillation.append(abs(t_x - t_xplus10))
    avg_oscilation = sum(oscillation) / len(oscillation)
    return avg_oscilation


def runode(x0, t, p, u):
    ans = odeint(evalf, x0, t, args=(p, u), full_output=True)
    return ans


def generate_inputs(n):
    """
    :param n (int): number of nodes
    :return: state vector x, parameters p, inputs u
    """
    # Placeholders
    y = np.ones([n])                    # n x 1
    tilde_y = np.ones([n])              # n x 1
    mu = np.ones([n])                   # n x 1
    tau1 = 1 * np.ones([n])             # n x 1
    tau2 = 1 * np.ones([n])             # n x 1
    sigma = 1e-3 * np.ones([n])         # n x 1
    alpha = -1*np.ones([n])             # n x 1
    gamma = np.ones([n])                # n x 1
    d = np.ones([n, n])                 # nm x 1
    g = np.ones([n])                    # n x 1 (little y)
    delt_w = np.zeros([n])              # n x 1
    # Build x, p, u arrays
    x = np.array([y, tilde_y, mu])
    p = {'tau1': tau1, 'tau2': tau2, 'sigma': sigma, 'alpha': alpha, 'gamma': gamma, 'd': d, 'g': g}
    u = np.array(delt_w)
    return x, p, u

  
# test cases, all with two countries:
# 1) all equal (GF)
# 2) all equal, but y_tilde starts too high
# 3) start with slightly different currency values, should converge (JR)
# 4) test with very small tau2 / tau_mu (JR)
# 5) with large alpha, shouldn't ring (JR)
# 6) with small alpha, should ring (JR)
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
    #ans = runode(x0, t)[0]
    def f_wrapper(x, t):
        return evalf(x, t, p, u)
    def g_wrapper(x, t):
        g = evalg(x, t, p, u)[:]
        return g
    G = evalf(x0, t, p, u)
    ans = sdeint.itoint(f_wrapper, g_wrapper, x0, t)
    ans = einops.rearrange(ans, "t (d n) -> d n t", d=3)
    #print(F)
    #print(ans)
    plot_evolution(ans)

