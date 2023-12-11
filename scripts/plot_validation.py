import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import einops
import math


from domain_specific.x0 import generate_demo_inputs
from dynamic import explicit


countries = {0: 'USA', 1: 'EUR', 2: 'IND', 3: 'CHN', 4: 'MEX', 5: 'CAN', 6: 'BRA', 7: 'SGP', 8: 'AUS', 9: 'GHA'}


def plot_data(data):
    smoothed = moving_average(data['Price'], 15)
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
    smoothed = transform_data(smoothed)
    xs = np.array(list(explicit.simulate(**kwargs)))
    stacked = einops.rearrange(xs, 't (d c) -> c d t', d=3)
    plt.plot(np.exp(stacked[2, 0, 4950:5147]), color='green', label='Predicted')
    plt.plot(data['Date'], smoothed, color='green', label='Actual', linestyle='dashed')
    plt.axvline(x='06/22/2023', alpha=0.5, color='grey')
    plt.axvline(x='09/08/2023', alpha=0.5, color='grey')
    print((data[data['Date'] == '09/08/2023']['Price'].item() - data[data['Date'] == '06/22/2023']['Price'].item()) /
          data[data['Date'] == '06/22/2023']['Price'].item())
    print(np.max(np.exp(stacked[2, 0, 5147]) - np.exp(stacked[2, 0, 4950])))
    xticks = plt.gca().xaxis.get_major_ticks()
    for i in range(len(xticks)):
        if i % 10 != 0:
            xticks[i].set_visible(False)
    plt.xticks(rotation=90)
    plt.ylabel("$y$")
    plt.legend()
    plt.show()


def transform_data(smoothed):
    max = np.max([x for x in smoothed if not math.isnan(x)])
    # Fit starting currency value to moving average
    return (smoothed / max) - 0.015


def moving_average(data, window_size):
    """
    Apply a simple moving average to smooth the input data.

    Parameters:
    - data: The input data array.
    - window_size: The size of the moving average window.

    Returns:
    - smoothed_data: The smoothed data array.
    """
    weights = np.repeat(1.0, window_size) / window_size
    smoothed_data = np.convolve(data, weights, 'valid')
    # Pad the beginning of the smoothed data with NaN values
    smoothed_data = np.concatenate([np.full(window_size - 1, np.nan), smoothed_data])
    return smoothed_data


if __name__ == "__main__":
    data = pd.read_csv('./scripts/india.csv')[::-1]
    plot_data(data)

