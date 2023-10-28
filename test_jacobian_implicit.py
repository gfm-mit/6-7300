from jacobian_implicit import tgcr, jf_product
from test_evalf import evalf, generate_inputs
from jacobian import evalJacobian
from sys import getsizeof
import matplotlib.pyplot as plt
import time


def measure_mem():
    t = None
    f_size, J_size = [], []
    for i in range(2, 100):
        x, p, u = generate_inputs(i)
        x = x.reshape(-1, )
        # Size of output of f (x2?)
        f = evalf(x, t, p, u)
        f_size.append(getsizeof(f) * 2)
        # Size of Jacobian
        J = evalJacobian(x, p, u)
        J_size.append(getsizeof(J))
    plt.plot(f_size, label="Implicit Jacobian")
    plt.plot(J_size, label="Explicit Jacobian")
    plt.legend()
    plt.xlabel("Size of input")
    plt.ylabel("Memory (bytes)")
    plt.title("Memory improvement using implicit Jacobian")
    plt.savefig('implicit_jacobian_mem.png', bbox_inches='tight')
    return


def measure_speed():
    t = None
    f_time, J_time = [], []
    for i in range(2, 100):
        x, p, u = generate_inputs(i)
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
    plt.plot(f_time, label="Implicit Jacobian")
    plt.plot(J_time, label="Explicit Jacobian")
    plt.legend()
    plt.xlabel("Size of input")
    plt.ylabel("Time to compute Jacobian (s)")
    plt.title("Speed improvement using implicit Jacobian")
    plt.savefig('implicit_jacobian_speed.png', bbox_inches='tight')
    return


if __name__ == '__main__':
    #measure_mem()
    measure_speed()
