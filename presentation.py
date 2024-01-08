import cartopy.crs as ccrs
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import subprocess

import sys
import os
import pathlib

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
from dynamic import explicit
import newton.from_julia
import domain_specific.demo
import viz.trajectory
import viz.animation
import scripts.plot_validation

# generate time track
# and then cool map
# validation plot

def cue_card(text):
    plt.figure(figsize=(8, 6))
    plt.axis('off')

    # Add text to the figure
    plt.text(0.5, 0.5, text, fontsize=40, ha='center', va='center', wrap=True)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Generate data to visualize
    x0, p, u = domain_specific.demo.generate_wobble_inputs(10)
    #np.random.seed()
    #cue_card("Ready to Begin Simulation")
    #viz.trajectory.plot_trajectory(p, u, x0)
    #cue_card("Movie Rendering is Too Slow to do in Real Time")
    ##viz.animation.plot_animation(p, u, x0, one_frame=False)
    #movie_file = 'animation.mp4'
    #subprocess.run(['open', movie_file])
    scripts.plot_validation.run_validation()