"""
Creates correlation plots
"""
# Standard packages

# Third-party packages
import numpy as np
import matplotlib.pyplot as plt

# Local packages
from correlation_matrix import correlation_matrix

def correlation_plot(bands, nodata_val = None, band_names = None) -> dict:
    """
    Plots correlation for input bands
    :param band: bands to plot data for
    :param nodata_val: No data value for given band
    :param band_names: Names of the bands to be plotted
    """
    n_bands = bands.shape[2]

    # Flatten out data so it can be plotted easier
    bands_reshaped = bands.reshape(-1, n_bands)

    # Mask out the nodata values
    masked_data = np.ma.masked_where(bands_reshaped == nodata_val, bands_reshaped)

    # Create pairwise vector subplots
    vec_fig, vec_axs = plt.subplots(n_bands, n_bands, sharex=True, sharey=True)
    for r_idx in range(n_bands):
        for c_idx in range(n_bands):
            vec_axs[r_idx, c_idx].scatter(masked_data[:,c_idx], masked_data[:,r_idx], s=1.0)

    # Give bands names if they aren't provided
    if band_names is None:
        band_names = range(n_bands)

    # Label cols
    for col_ax, band_name in zip(vec_axs[0], band_names):
        col_ax.set_title(f"Band {band_name}")

    # Label rows
    for row_ax, band_name in zip(vec_axs[:,0], band_names):
        row_ax.set_ylabel(f"Band {band_name}")

    vec_fig.tight_layout()
    plt.show()

    # Create density plots
    den_fig, den_axs = plt.subplots(n_bands, n_bands, sharex=True, sharey=True)
    for r_idx in range(n_bands):
        for c_idx in range(n_bands):
            # Create color for plot
            den_axs[r_idx, c_idx].hist2d(masked_data[:,c_idx], masked_data[:,r_idx], bins=(50,50), cmap='jet')

    # Label cols
    for col_ax, band_name in zip(den_axs[0], band_names):
        col_ax.set_title(f"Band {band_name}")

    # Label rows
    for row_ax, band_name in zip(den_axs[:,0], band_names):
        row_ax.set_ylabel(f"Band {band_name}")

    den_fig.tight_layout()
    plt.show()

    corr_matrix = correlation_matrix(bands, 0)

    return corr_matrix

if __name__ == "__main__":
    # Define path for data and load in array
    path = "eda/sentinel2_rochester.npy"
    bands = np.load(path)

    # Define band names and wavelengths according to sentinel-2 website
    sentinel2_band_names = [
        "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8a", "B9", "B11", "B12"
    ]

    band_idxs = [1, 2, 3, 7]
    band_names = [sentinel2_band_names[idx] for idx in band_idxs]

    corr_matrix = correlation_plot(bands[:,:,band_idxs], 0, band_names)
