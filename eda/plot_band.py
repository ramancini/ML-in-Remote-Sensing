"""
Function to plot a single band of data
"""
# Standard packages

# Third-party packages
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np

# Local packages

def plot_band(band, name, wavelength, nodata_val = None, cmap = cm.grey) -> bool:
    """
    Extracts and plots bands from a dataset.
    :param band: Numpy array containing a single spectral band
    :param name: Name for the spectral band being displayed
    :param wavelength: Wavelength of the band being displayed
    :param nodata_val: The no-data value in the band
    :return: True upon successful run of the function
    """
    # Make a copy of band
    tmp_band = band

    # Make nodata value black if provided
    if nodata_val is not None:
        tmp_band = np.ma.masked_where(tmp_band == nodata_val, tmp_band)
        cmap.set_bad(color='black')

    # Plot band
    plt.imshow(tmp_band, cmap=cmap)
    plt.title(f"{name} (\lambda = {wavelength})")
    plt.show()

    return True

if __name__ == "__main__":
    # Define path for data and load in array
    path = "eda/sentinel2_rochester.npy"
    bands = np.load(path)

    # Define band names and wavelengths according to sentinel-2 website
    sentinel2_band_names = [
        "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8a", "B9", "B11", "B12"
    ]
    sentinel2_band_wavelengths = [
        442.7, 492.7, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 945.1, 1613.7, 2202.4
    ]

    # Display each band
    for idx in range(bands.shape[2]):
        plot_band(bands[:,:,idx], sentinel2_band_names[idx], sentinel2_band_wavelengths[idx], 0)