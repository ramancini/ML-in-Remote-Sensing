"""

"""
import matplotlib.pyplot as plt
# Standard packages

# Third-party packages
import numpy as np
from numpy.linalg import norm
import pandas as pd

# Local packages

def sam(v1, v2):
    """
    Calculate the cosine similarity between two vectors.
    :param v1: First input vector
    :param v2: Second input vector
    :return: Angle
    """
    angle = np.arccos(np.dot(v1, v2) / (norm(v1, axis=-1) * norm(v2, axis=-1)))

    return angle


if __name__ == "__main__":
    # Define path for data and load in array
    band_path = "eda/sentinel2_rochester.npy"
    bands = np.load(band_path)

    oak_path = "eda/oak.csv"
    road_path = "eda/road.csv"

    oak_data = pd.read_csv(oak_path)
    road_data = pd.read_csv(road_path)

    # Define band names and wavelengths according to sentinel-2 website
    sentinel2_band_names = [
        "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8a", "B11", "B12"
    ]
    sentinel2_band_wavelengths = [
        0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 1.610, 2.190
    ]
    sentinel2_band_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]

    # Interpolate oak and road data to match sentinel data
    oak_interp = np.interp(sentinel2_band_wavelengths,
                           oak_data['wavelength'],
                           oak_data['reflectance']) / 100
    road_interp = np.interp(sentinel2_band_wavelengths,
                            road_data['wavelength'],
                            road_data['reflectance']) / 100

    # Calculate angle between oak and every band / pixel
    oak_angle = sam(bands[:,:,sentinel2_band_idxs], oak_interp)
    road_angle = sam(bands[:,:,sentinel2_band_idxs], road_interp)

    # Find 1st, 50th, and 100th closest values
    oak_sort_idxs = np.argsort(oak_angle, None)
    road_sort_idxs = np.argsort(road_angle, None)

    oak_close_idxs_flat = oak_sort_idxs[[0,49,99]]
    road_close_idxs_flat = road_sort_idxs[[0,49,99]]

    # Need to "unflatten" indices
    oak_close_idxs = np.unravel_index(oak_close_idxs_flat, oak_angle.shape)
    road_close_idxs = np.unravel_index(road_close_idxs_flat, road_angle.shape)

    # Plot original oak data alongside 1st, 50th, and 100th closest match
    plt.plot(oak_data['wavelength'], oak_data['reflectance'] / 100,
             label='ECOSTRESS Oak Data')
    plt.plot(sentinel2_band_wavelengths,
             bands[oak_close_idxs[0][0],
                   oak_close_idxs[1][0],
                   sentinel2_band_idxs],
             label=f"1st closest match "
                   f"({oak_close_idxs[0][0]},{oak_close_idxs[1][0]})")
    plt.plot(sentinel2_band_wavelengths,
             bands[oak_close_idxs[0][1],
                   oak_close_idxs[1][1],
                   sentinel2_band_idxs],
             label=f"50th closest match "
                   f"({oak_close_idxs[0][1]},{oak_close_idxs[1][1]})")
    plt.plot(sentinel2_band_wavelengths,
             bands[oak_close_idxs[0][2],
             oak_close_idxs[1][2],
             sentinel2_band_idxs],
             label=f"100th closest match "
                   f"({oak_close_idxs[0][2]},{oak_close_idxs[1][2]})")
    plt.xlim((0.35, 2.5))
    plt.ylabel("Reflectance")
    plt.xlabel("Wavelength [microns]")
    plt.title("Oak Reflectance Data")
    plt.legend()
    plt.show()

    # Plot original road data alongside 1st, 50th, and 100th closest match
    plt.plot(road_data['wavelength'], road_data['reflectance'] / 100,
             label='ECOSTRESS Road Data')
    plt.plot(sentinel2_band_wavelengths,
             bands[road_close_idxs[0][0],
             road_close_idxs[1][0],
             sentinel2_band_idxs],
             label=f"1st closest match "
                   f"({road_close_idxs[0][0]},{road_close_idxs[1][0]})")
    plt.plot(sentinel2_band_wavelengths,
             bands[road_close_idxs[0][1],
             road_close_idxs[1][1],
             sentinel2_band_idxs],
             label=f"50th closest match "
                   f"({road_close_idxs[0][1]},{road_close_idxs[1][1]})")
    plt.plot(sentinel2_band_wavelengths,
             bands[road_close_idxs[0][2],
             road_close_idxs[1][2],
             sentinel2_band_idxs],
             label=f"100th closest match "
                   f"({road_close_idxs[0][2]},{road_close_idxs[1][2]})")
    plt.xlim((0.35, 2.5))
    plt.ylabel("Reflectance")
    plt.xlabel("Wavelength [microns]")
    plt.title("Road Reflectance Data")
    plt.legend()
    plt.show()

    # Determine a threshold, we'll exclude anything with cosine similarity
    # under 0.97
    oak_fig, oak_axs = plt.subplots(3, 4, sharex=True, sharey=True)
    for idx, band_idx in enumerate(sentinel2_band_idxs):
        ridx = idx // 4
        cidx = idx % 4

        masked_band = np.zeros(bands[:,:,band_idx].shape)
        # Higher similarity = lower angle so we use less than or equal to
        masked_band[oak_angle <= np.arccos(0.97)] = 1

        oak_axs[ridx, cidx].imshow(masked_band)
        oak_axs[ridx, cidx].set_title(f"Oak thresholded band {sentinel2_band_names[idx]}")

    plt.show()

    road_fig, road_axs = plt.subplots(3, 4, sharex=True, sharey=True)
    for idx, band_idx in enumerate(sentinel2_band_idxs):
        ridx = idx // 4
        cidx = idx % 4

        masked_band = np.zeros(bands[:, :, band_idx].shape)
        # Higher similarity = lower angle so we use less than or equal to
        masked_band[road_angle <= np.arccos(0.97)] = 1

        road_axs[ridx, cidx].imshow(masked_band)
        road_axs[ridx, cidx].set_title(
            f"Road thresholded band {sentinel2_band_names[idx]}")

    plt.show()
