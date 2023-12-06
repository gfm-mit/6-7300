import matplotlib.pyplot as plt

from dynamic import explicit, implicit
from utils.performance import measure_speed, measure_mem


def plot_mem():
    f1_size, f2_size, f1_peak, f2_peak = measure_mem(range(2, 100))
    color = plt.plot(f1_size, label="Trapezoidal")[0].get_color()
    plt.plot(f1_peak, color=color, dashes=[1,1], zorder=20)
    color = plt.plot(f2_size, label="RK4")[0].get_color()
    plt.plot(f2_peak, color=color, dashes=[1,1], zorder=20)
    plt.legend()
    plt.xlabel("Size of input")
    plt.ylabel("Memory (bytes)")
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Memory improvement using implicit Jacobian")
    return


def plot_speed():
    evaltrap_kwargs = {'x': None, 'p': None, 'u': None, 't': 40, 'delta_t': 1e-5, 'f_step': implicit.get_trapezoid_f}
    evalrk4_kwargs = {'x': None, 'p': None, 'u': None, 't': 40, 'delta_t': 1e-5, 'f_step': explicit.rk4}
    trap_time, rk4_time = measure_speed(range(2, 100), implicit.simulate, evaltrap_kwargs, explicit.simulate, evalrk4_kwargs)
    plt.plot(trap_time, label="Trapezodial")
    plt.plot(rk4_time, label="RK4")
    plt.legend()
    plt.xlabel("Size of input")
    plt.ylabel("Time to run simulation (s)")
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Speed improvement using RK4")
    return


if __name__ == '__main__':
    plot_speed()
    plt.savefig('time_integrator_speed.png', bbox_inches='tight')
    plt.show()
    plot_mem()
    plt.savefig('time_integrator_mem.png', bbox_inches='tight')
    plt.show()
