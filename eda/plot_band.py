"""
Function to plot 12 bands from the Sentinel-2 dataset
"""
# Standard packages

# Third-party packages
import numpy as np

# Local packages

def plot_band(path) -> bool:
    """
    Extracts and plots bands from Sentinel-2 dataset. Band 10 should be excluded from the input data
    :param path: Path to the numpy file to import the data from
    :return: True upon successful run of the function
    """

    return True

if __name__ == "__main__":
    path = "eda/"
    plot_band()