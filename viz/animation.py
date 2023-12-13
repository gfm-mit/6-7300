import cartopy.crs as ccrs
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np

import sys
import os
import pathlib

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
from dynamic import explicit
import newton.from_julia
import domain_specific.demo

def setup():
    # Set up map
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.RotatedPole(
        #central_latitude=45,
        #central_longitude=20,
        pole_latitude=55,
        pole_longitude=-170,
    )})
    ax.stock_img()

    return fig, ax


def render_frame(frame_idx, p, u, xs, locs, ax, xms):
    T = frame_idx * 333
    ax.clear()
    ax.stock_img()
    p_demo = p.copy()
    p_demo['d'] = p_demo['d'][T, :, :]
    print("T:", T)
    state = xs[T]
    x_ij = xms[T]
    sizes = 300 * np.exp(1 * (state - xs[0]))
    for idx, row in locs.iterrows():
        size = sizes[idx]
        # Plot nodes
        # Get currency value from country idx at time t
        # Convert from log
        color = "yellowgreen"
        if idx == 0:
            color = "mediumblue"
        elif idx == 2:
            color = "goldenrod"
        ax.scatter(row['long'], row['lat'],
                   s=size+100, color="dimgray", 
                   transform=ccrs.PlateCarree(), zorder=2)
        ax.scatter(row['long'], row['lat'],
                   s=size, color=color, 
                   transform=ccrs.PlateCarree(), zorder=10)
        ax.text(row['long']-10, row['lat']+5,
                row['country'],
                ha='right',
                va='bottom',
                transform=ccrs.PlateCarree(), zorder=11)
        # Plot edges
        if idx not in [0, 2]:
            continue
        for idx_other, row_other in locs.iterrows():
            if idx_other == idx:
                continue
            if idx_other == 2:
                continue
            # Exports
            coords = np.stack([
                np.linspace(row['long'], row_other['long'], 100),
                np.linspace(row['lat'], row_other['lat'], 100)
            ]).transpose()

            e_x = x_ij[idx, idx_other] / xms[2, idx, idx_other]
            e_m = x_ij[idx_other, idx] / xms[2, idx_other, idx]
            x_width = 40. * np.maximum(0., .1 + np.log(e_x))
            m_width = 40. * np.maximum(0., .1 + np.log(e_m))


            segments = np.stack([coords[:-1], coords[1:]], axis=1)
            lwidths = np.linspace(x_width, m_width, segments.shape[0])
            start_color = color
            end_color = "limegreen" if idx_other != 0 else "mediumblue"

            # Generate ten colors interpolating between the start and end colors
            gradient = [mcolors.to_hex(c)
                      for c in mcolors.LinearSegmentedColormap.from_list(
                          "", [start_color, end_color])(
                              np.linspace(0, 1, segments.shape[0]) ** 2)]

            lc = LineCollection(segments, linewidths=lwidths,color=gradient, zorder=0, transform=ccrs.PlateCarree())

            ax.add_collection(lc)
    return


def plot_animation(p, u, x0, one_frame=True):
    p_initial = p.copy()
    p_initial['d'] = p_initial['d'][0, :, :]
    x1 = newton.from_julia.newton_julia_jacobian_free_wrapper(x0, p_initial, u)
    t1 = 100
    kwargs = dict(
        x0=x1,
        p=p,
        u=u,
        t1=t1,
        delta_t=1e-2,
        f_step=explicit.rk4,
        demo=True
    )
    xs = np.array(list(explicit.simulate(**kwargs)))
    xms = np.stack(list(explicit.simulate_exports(xs, **kwargs)))

    fig, ax = setup()
    locs = pd.read_csv('./viz/data.csv')
    if one_frame:
        render_frame(30, p, u, xs, locs, ax, xms)
        plt.show()
    else:
        ani = FuncAnimation(fig, render_frame, frames=range(30), fargs=(
            p, u, xs, locs, ax, xms,
            ), interval=10000)
        ani.save('animation.mp4', fps=10)

if __name__ == "__main__":
    # Generate data to visualize
    x0, p, u = domain_specific.demo.generate_wobble_inputs(10)
    plot_animation(p, u, x0, one_frame=False)