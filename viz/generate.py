import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np

from dynamic import explicit, implicit
from domain_specific.x0 import generate_stochastic_inputs


def setup(locs):
    # Set up map
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    ax.stock_img()

    # Plot relevant countries as nodes on map
    for idx, row in locs.iterrows():
        plt.plot(row['long'], row['lat'], 'o', transform=ccrs.PlateCarree())
        plt.text(row['long'], row['lat'], row['country'], transform=ccrs.PlateCarree())

    return fig, ax


def step(t, xs, locs):
    for idx, row in locs.iterrows():
        size = xs[t][idx] # Get currency value from country idx at time t
        plt.plot(row['long'], row['lat'], 'o', markersize=size, transform=ccrs.PlateCarree())
        plt.text(row['long'], row['lat'], row['country'], transform=ccrs.PlateCarree())
    return


def animate(fig, xs, locs):
    ani = FuncAnimation(fig, step, fargs=(xs, locs), interval=100)
    return ani


if __name__ == "__main__":
    # Setup map
    locs = pd.read_csv('./viz/data.csv')
    fig, ax = setup(locs)

    # Generate data to visualize
    # TODO: change to reading saved npy file
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

    ani = animate(fig, xs, locs)
    ani.save('animation.mp4')

    plt.show()
