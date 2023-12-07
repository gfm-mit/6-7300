import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
import os

from dynamic import explicit, implicit
from domain_specific.x0 import generate_stochastic_inputs


def setup():
    # Set up map
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    ax.stock_img()

    return fig, ax


def step(t, xs, locs, ax):
    ax.clear()
    ax.stock_img()
    for idx, row in locs.iterrows():
        size = abs(xs[t, idx]) # Get currency value from country idx at time t
        ax.plot(row['long'], row['lat'], 'o', markersize=100*size, transform=ccrs.PlateCarree())
        ax.text(row['long'], row['lat'], row['country'], transform=ccrs.PlateCarree())
    return


def animate(fig, xs, locs, ax):
    ani = FuncAnimation(fig, step, frames=range(100), fargs=(xs, locs, ax), interval=100)
    return ani


if __name__ == "__main__":
    # Setup map
    locs = pd.read_csv('./viz/data.csv')
    fig, ax = setup()

    # Generate data to visualize
    if os.path.exists('./viz/xs.npy'):
        xs = np.load('./viz/xs.npy')
    else:
        x0, p, u = generate_stochastic_inputs(3)
        kwargs = dict(
            x0=x0,
            p=p,
            u=u,
            t1=1,
            delta_t=1e-2,
            f_step=explicit.rk4,
        )
        xs = np.array(list(explicit.simulate(**kwargs)))
        np.save('./viz/xs.npy', xs)

    ani = animate(fig, xs, locs, ax)
    ani.save('animation.mp4')

    plt.show()
