"""
Calculate the Pearson r correlation coefficient matrix
"""
# Standard packages

# Third-party packages
from matplotlib import pyplot as plt
import numpy as np

# Local packages

def correlation_matrix(bands, nodata_val = None):
    """
    Calculates the correlation coefficient matrix for a numpy array of multispectral band data
    :param bands: bands to use for calculation of coefficient matrix
    :param nodata_val: No data value for given band
    """
    # Reshape the bands so we can run it through numpy corrcoeff
    reshaped_data = np.transpose(bands.reshape(-1, bands.shape[2]))

    # Mask out the nodata values
    masked_data = np.ma.masked_where(reshaped_data == nodata_val, reshaped_data)

    # Calculate correlation coefficient
    coeff_mat = np.corrcoef(masked_data)

    return coeff_mat

if __name__ == "__main__":
    # Define path for data and load in array
    path = "eda/sentinel2_rochester.npy"
    bands = np.load(path)

    # Define band names and wavelengths according to sentinel-2 website
    sentinel2_band_names = [
        "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8a", "B9", "B11", "B12"
    ]

    output = correlation_matrix(bands, 0)

    plt.imshow(output, cmap='grey')
    plt.colorbar()
    plt.title("Pearson R Correlation Coefficients")
    plt.show()
