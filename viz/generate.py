import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np

from dynamic import explicit
from domain_specific.x0 import generate_stochastic_real_inputs, generate_demo_inputs
from domain_specific.evalf import evalf


def setup():
    # Set up map
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    ax.stock_img()

    return fig, ax


def step(t, p, u, xs, locs, ax):
    ax.clear()
    ax.stock_img()
    p_demo = p.copy()
    p_demo['d'] = p_demo['d'][t, :, :]
    _, x_ij = evalf(xs[t], None, p_demo, u, yield_intermediates=True)
    for idx, row in locs.iterrows():
        # Plot nodes
        # Get currency value from country idx at time t
        # Convert from log
        size = np.exp(xs[t, idx]) / np.exp(xs[t]).max()
        ax.plot(row['long'], row['lat'], 'o', markersize=15 * size, color='blue', transform=ccrs.PlateCarree(),
                zorder=2)
        ax.text(row['long']+5, row['lat'], row['country'], transform=ccrs.PlateCarree(), zorder=3)
        # Plot edges
        for idx_other, row_other in locs.iterrows():
            if idx_other != idx:
                # Exports
                n = x_ij[idx, idx_other] / x_ij.max()
                n_other = x_ij[idx_other, idx] / x_ij.max()
                color = "green"
                ax.plot([row['long']+1, row_other['long']+1], [row['lat']+1, row_other['lat']+1],
                        linewidth=5*abs(n), color=color, alpha=0.5, transform=ccrs.PlateCarree(), zorder=1)
                ax.plot([row['long']-1, row_other['long']-1], [row['lat']-1, row_other['lat']-1],
                        linewidth=5*abs(n_other), color=color, alpha=0.5, transform=ccrs.PlateCarree(), zorder=1)
    return


def animate(fig, p, u, xs, locs, ax):
    ani = FuncAnimation(fig, step, frames=range(10000), fargs=(p, u, xs, locs, ax), interval=10000)
    return ani


if __name__ == "__main__":
    # Setup map
    locs = pd.read_csv('./viz/data.csv')
    fig, ax = setup()

    # Generate data to visualize
    x0, p, u = generate_demo_inputs(10)
    kwargs = dict(
        x0=x0,
        p=p,
        u=u,
        t1=100,
        delta_t=1e-2,
        f_step=explicit.rk4,
        demo=True
    )
    xs = np.array(list(explicit.simulate(**kwargs)))

    ani = animate(fig, p, u, xs, locs, ax)
    ani.save('animation.mp4', fps=100)

    plt.show()
