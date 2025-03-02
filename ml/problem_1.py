"""
Focuses on displaying the data for problem 1 in ML homework 2
"""
# Standard packages

# Third-party packages
from matplotlib import pyplot as plt
import spectral

# Local packages
from eda.correlation_matrix import correlation_matrix

if __name__ == '__main__':
    # Load in the envi data
    packed_path = "ml/materials/tait_hsi"

    packed_data = spectral.envi.open(packed_path + ".hdr", packed_path)
    data = packed_data.load()

    # Display red, green, and blue bands
    # 630nm = red = 104
    # 532nm = green = 60
    # 465nm = blue = 30
    truecolor_img = spectral.get_rgb(data, (104, 60, 30))
    plt.imshow(truecolor_img)
    plt.title("Truecolor Image (630.075nm, 532.132nm, 465.352nm)")
    plt.show()

    # 810nm = NIR = 185
    # Show a pseudocolor image with red = 810nm, green = 630nm, blue = 532nm
    pscolor_img = spectral.get_rgb(data, (185, 104, 60))
    plt.imshow(pscolor_img)
    plt.title("Pseudocolor image R=810.379nm, G=630.075nm, B=532.132nm")
    plt.show()

    # Calculate and plot correlation matrix
    mx1_corr_mat = correlation_matrix(data)
    plt.imshow(mx1_corr_mat, cmap='gray')
    plt.title("Correlation matrix for Tait reserve data")
    plt.colorbar()
    plt.show()