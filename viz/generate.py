import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd


def generate_map(loc_data):
    # Set up map
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    ax.stock_img()

    # Plot relevant countries as nodes on map
    locations = pd.read_csv(loc_data)
    for idx, row in locations.iterrows():
        plt.plot(row['long'], row['lat'], 'o', transform=ccrs.PlateCarree())
        plt.text(row['long'], row['lat'], row['country'], transform=ccrs.PlateCarree())

    return fig, ax


if __name__ == "__main__":
    fig, ax = generate_map('./viz/data.csv')
    plt.show()
