import numpy as np
import matplotlib.pyplot as plt
import einops

from domain_specific.x0 import generate_stochastic_real_inputs, generate_stochastic_inputs
from dynamic import explicit


def visualize_real(n, t1=40):
    x0, p, u = generate_stochastic_real_inputs(n)
    kwargs = dict(
        x0=x0,
        p=p,
        u=u,
        t1=t1,
        delta_t=1e-2,
        f_step=explicit.rk4,
    )
    xs = np.array(list(explicit.simulate(**kwargs)))
    stacked = einops.rearrange(xs, 't (d c) -> c d t', d=3)
    for i in range(n):
        plt.plot(stacked[i, 0], label=f"Country {i}")
    plt.show()
    return


if __name__ == '__main__':
    n = 10
    visualize_real(n)
