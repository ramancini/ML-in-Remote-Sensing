"""
Function for performing principal component analysis
"""
# Standard packages

# Third-party packages
from matplotlib import pyplot as plt
import numpy as np
from scipy.linalg import svd
import spectral

# Local packages

def principal_component_analysis(array):
    """
    Takes an array and calculates the principal components, eigenvalues, and
    standardizes the data.
    :param array: Should be of size m*d where m is the number of samples and d
                  is the number of features.
    """
    # Standardize the data
    mean_arr = (array - np.mean(array, axis=0)) / np.std(array, axis=0)

    # Perform SVD - scipy seems to sort the pcs already by their eigen values
    u_mat, singular_vals, vt_mat = svd(mean_arr, full_matrices=False)
    pcs = vt_mat.T

    # Should be faster than np.pow
    eigenvalues = (singular_vals * singular_vals) / (array.shape[0] - 1)

    return pcs, eigenvalues, mean_arr

if __name__ == '__main__':
    # Load in data
    tait_data_path = '/Users/ramancini/src/python/ML-in-Remote-Sensing/ml/materials/tait_hsi'
    packed_data = spectral.envi.open(tait_data_path + '.hdr', tait_data_path)
    data = packed_data.load()

    # Grab the wavelengths for later
    wavelengths = [float(n.split()[0]) for n in data.metadata["band names"]]

    # Reshape the data and perform pca
    data_reshape = data.reshape([data.shape[0] * data.shape[1], data.shape[2]])
    tait_pcs, tait_evals, tait_mean_arr = principal_component_analysis(data_reshape)

    # Select the first 10 principal components
    first_pcs = tait_pcs[:,:10]

    # Reproject the data
    reproj_data = np.dot(tait_mean_arr, first_pcs)

    # Reshape the data back to original shape
    reproj_data_reshape = reproj_data.reshape([data.shape[0], data.shape[1], -1])

    # Plot
    fig, axs = plt.subplots(2, 5, sharey=True)
    for idx in range(10):
        ridx = idx // 5
        cidx = idx % 5

        img = axs[ridx, cidx].imshow(reproj_data_reshape[:,:,idx], cmap='gray')
        axs[ridx, cidx].set_title(f"Principal Component {idx + 1}")
        fig.colorbar(img, ax=axs[ridx, cidx], shrink=0.5)

    plt.show()

    # L2 Reprojection error
    pc_idx_list = [1,10,50,100,tait_mean_arr.shape[1]]

    l2_dist_list = []
    for idx in pc_idx_list:
        # Select first N PCs
        select_pcs = tait_pcs[:,:idx]

        # Perform reconstruction
        reduced_data = np.dot(tait_mean_arr, select_pcs)
        reconstruct_data = np.dot(reduced_data, select_pcs.T)

        # Calculate L2 dist
        l2_dist_diff = tait_mean_arr - reconstruct_data
        l2_dist = np.sqrt(np.sum(l2_dist_diff * l2_dist_diff, axis=1))
        avg_l2_dist = np.mean(l2_dist)

        # Add to list of L2 distances to plot
        l2_dist_list.append(avg_l2_dist)

    # Plot the L2 distances
    plt.plot(pc_idx_list, l2_dist_list)
    plt.xlim([1, pc_idx_list[-1]])
    plt.xlabel("Number of Principal Components Selected")
    plt.ylabel("Average L2 Distance Error")
    plt.title("Mean Reconstruction Error for Tait Reserve PCA")
    plt.show()

    # Compute explained variance ratio for each eigenvalue
    ex_var_ratio = tait_evals / np.sum(tait_evals)
    cumulative_var_ratio = np.cumsum(ex_var_ratio)

    # Set PCs with cumulative var under 99 to 0
    var_select_pcs = tait_pcs.copy()
    var_select_pcs[:, cumulative_var_ratio > 0.99] = 0

    # Apply inverse PCA transform
    reduced_arr = np.dot(tait_mean_arr, tait_pcs)
    inverse_arr = (np.dot(reduced_arr, var_select_pcs.T) * np.std(data_reshape, axis=0)) + np.mean(data_reshape, axis=0)

    # Reshape the data to make pixels easier to grab
    inverse_arr_reshape = inverse_arr.reshape([data.shape[0], data.shape[1], -1])

    #                x,   y
    # Dock =       996, 750
    # Gravel =     603, 596
    # Vegetation = 898, 806
    # Dirt =       618,  57
    # Water =      445, 625
    # Plot 5 interesting pixels
    px_fig, px_axs = plt.subplots(2, 3, sharex=True)

    px_axs[0, 0].plot(wavelengths, data[750,996,:].flatten())
    px_axs[0, 0].plot(wavelengths, inverse_arr_reshape[750, 996, :].flatten())
    px_axs[0, 0].set_title("Dock Spectral Data [996,750]")
    px_axs[0, 0].set_ylim([0,6])

    px_axs[0, 1].plot(wavelengths, data[596, 603, :].flatten())
    px_axs[0, 1].plot(wavelengths, inverse_arr_reshape[596, 603, :].flatten())
    px_axs[0, 1].set_title("Gravel Spectral Data [603,596]")
    px_axs[0, 1].set_ylim([0, 6])

    px_axs[0, 2].plot(wavelengths, data[806, 898, :].flatten())
    px_axs[0, 2].plot(wavelengths, inverse_arr_reshape[806, 898, :].flatten())
    px_axs[0, 2].set_title("Vegetation Spectral Data [898,806]")
    px_axs[0, 2].set_ylim([0, 6])

    px_axs[1, 0].plot(wavelengths, data[57, 618, :].flatten())
    px_axs[1, 0].plot(wavelengths, inverse_arr_reshape[57, 618, :].flatten())
    px_axs[1, 0].set_title("Dirt Spectral Data [618,57]")
    px_axs[1, 0].set_ylim([0, 6])

    px_axs[1, 1].plot(wavelengths, data[625, 445, :].flatten())
    px_axs[1, 1].plot(wavelengths, inverse_arr_reshape[625, 445, :].flatten())
    px_axs[1, 1].set_title("Water Spectral Data [445,625]")
    px_axs[1, 1].set_ylim([0, 6])
    plt.show()

    # Mask out no-data values
    data_masked = np.ma.masked_where(data_reshape == 0, data_reshape)
    inverse_masked = np.ma.masked_where(data_reshape == 0, inverse_arr)

    # Calculate SNR
    snr_raw = np.mean(data_masked, axis=0) / np.std(data_masked, axis=0)
    snr_inverse = np.mean(inverse_masked, axis=0) / np.std(inverse_masked, axis=0)
    print(f"Average SNR Raw Image: {np.mean(snr_raw)}")
    print(f"Average SNR Inverse Image: {np.mean(snr_inverse)}")
