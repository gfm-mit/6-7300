import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_data(data):
    smoothed = moving_average(data['Price'], 7)
    plt.plot(data['Date'], data['Price'])
    plt.plot(data['Date'], smoothed)
    plt.axvline(x='09/08/2023', linestyle='--', color='grey')
    xticks = plt.gca().xaxis.get_major_ticks()
    for i in range(len(xticks)):
        if i % 10 != 0:
            xticks[i].set_visible(False)
    plt.xticks(rotation=90)
    plt.show()


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

