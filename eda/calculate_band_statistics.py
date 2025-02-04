"""
Calculates the statistics of a single band
"""
# Standard packages

# Third-party packages
import numpy as np
from scipy.stats import mstats

# Local packages

def calculate_band_statistics(band, nodata_val = None) -> dict:
    """
    Calculates the mean, standard deviation (std), minimum, maximum, quartiles (Q1, median, Q3), skewness, and kurtosi
    :param band: Single band of data to calculate stats for
    :param nodata_val: No data value for given band
    """
    # Mask out the nodata values
    masked_data = np.ma.masked_where(band == nodata_val, band)

    # Calculate statistics for array without nodata values
    data_mean = masked_data.mean()
    data_std = masked_data.std()
    data_min = masked_data.min()
    data_max = masked_data.max()
    data_median = np.ma.median(masked_data)
    data_q1 = np.nanpercentile(masked_data.filled(np.nan), 25)
    data_q3 = np.nanpercentile(masked_data.filled(np.nan), 75)
    data_skewness = mstats.skew(masked_data.flatten())
    data_kurtosis = mstats.kurtosis(masked_data.flatten())

    stats = {
        'mean': data_mean,
        'std': data_std,
        'min': data_min,
        'max': data_max,
        'Q1': data_q1,
        'median': data_median,
        'Q3': data_q3,
        'skewness': data_skewness,
        'kurtosis': data_kurtosis
    }

    return stats

if __name__ == "__main__":
    # Define path for data and load in array
    path = "eda/sentinel2_rochester.npy"
    bands = np.load(path)

    # Define band names and wavelengths according to sentinel-2 website
    sentinel2_band_names = [
        "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8a", "B9", "B11", "B12"
    ]

    for idx in range(bands.shape[2]):
        stats = calculate_band_statistics(bands[:,:,idx], 0)
        print(f"Band: {sentinel2_band_names[idx]}")
        print(f"  Mean:               {stats['mean']:0.3f}")
        print(f"  Median:             {stats['median']:0.3f}")
        print(f"  Standard Deviation: {stats['std']:0.3f}")
        print(f"  Minimum:            {stats['min']:0.3f}")
        print(f"  Maximum:            {stats['max']:0.3f}")
        print(f"  1st Quartile:       {stats['Q1']:0.3f}")
        print(f"  3rd Quartile:       {stats['Q3']:0.3f}")
        print(f"  Skewness:           {stats['skewness']:0.3f}")
        print(f"  Kurtosis:           {stats['kurtosis']:0.3f}")