"""
Calculates the z-score for a single multispectral band and normalizes data
"""
import matplotlib.pyplot as plt
# Standard packages

# Third-party packages
import numpy as np

# Local packages

def standardize(band, nodata_val = None):
    """
    Calculates the z-score for a multispectral band and normalizes it
    :param band: Single band of data to calculate stats for
    :param nodata_val: No data value for given band
    """
    # Mask out the nodata values
    masked_data = np.ma.masked_where(band == nodata_val, band)

    # Calculate statistics for array without nodata values
    data_mean = masked_data.mean()
    data_std = masked_data.std()

    # Calculate z-score for images
    data_standard = (masked_data - data_mean) / data_std

    return data_standard

if __name__ == "__main__":
    # Define path for data and load in array
    path = "eda/sentinel2_rochester.npy"
    bands = np.load(path)

    # Define band names and wavelengths according to sentinel-2 website
    sentinel2_band_names = [
        "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8a", "B9", "B11",
        "B12"
    ]

    fig, axs = plt.subplots(3, 4)
    for idx in range(bands.shape[2]):
        ridx = idx // 4
        cidx = idx % 4

        # Remove nodata values
        masked_data = np.ma.masked_where(bands[:,:,idx] == 0, bands[:,:,idx])

        axs[ridx, cidx].hist(masked_data.flatten(), bins=255)
        axs[ridx, cidx].set_title(f"{sentinel2_band_names[idx]} Histogram")
        axs[ridx, cidx].set_xlim((0, 1))

    plt.show()

    standard_fig, standard_axs = plt.subplots(3, 4)
    for idx in range(bands.shape[2]):
        ridx = idx // 4
        cidx = idx % 4

        # Standardize data
        standard_data = standardize(bands[:,:,idx], 0)

        standard_axs[ridx, cidx].hist(standard_data.flatten(), bins=255)
        standard_axs[ridx, cidx].set_title(f"{sentinel2_band_names[idx]} Histogram")
        standard_axs[ridx, cidx].axvline(-3, color='red', linewidth=0.5)
        standard_axs[ridx, cidx].axvline(3, color='red', linewidth=0.5)

    plt.show()