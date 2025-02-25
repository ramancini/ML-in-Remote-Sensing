"""
A python file
"""
# Standard packages

# Third-party packages
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import spectral

# Local packages
from ml.principal_component_analysis import principal_component_analysis

# TODO: vectorize to improve performance
def kmeans(data, k=5, max_iter=100):
    """
    Perform k-means clustering on data where data is an m*n matrix with m being
    the samples and n being the features
    :param data: m*n array where m is the samples and n is the features
    :param k: number of clusters
    :param max_iter: maximum number of iterations
    """
    # Calculate the minimum and maximum values for each feature
    min_vec = np.min(data, axis=0)
    max_vec = np.max(data, axis=0)

    k_ctrs = (np.random.rand(k, data.shape[1]) * (max_vec - min_vec)) + min_vec

    # Start iteration process
    curr_iter = 0
    prev_ctrs = np.zeros(k_ctrs.shape)
    while np.not_equal(prev_ctrs, k_ctrs).any() and curr_iter < max_iter:
        # Calculate distance from each point to each center
        k_ctrs_reshape = k_ctrs.reshape([k_ctrs.shape[0], 1, k_ctrs.shape[1]])
        pt_dists = np.sum(np.pow(k_ctrs_reshape - data, 2), axis=2)

        # Find which centroid is closest to every point using idxs as labels
        pt_close = np.argmin(pt_dists, axis=0)

        # Save k_ctrs before changing them
        prev_ctrs = k_ctrs.copy()

        for idx in range(k):
            data_mask = data[pt_close == idx, :]

            # Sometimes length can be 0, if so just leave the previous center
            if data_mask.shape[0] != 0:
                k_ctrs[idx] = np.mean(data_mask, axis=0)

        curr_iter += 1

    return pt_close


if __name__ == '__main__':
    # Load in jellybeans.tiff
    jellybean_path = "/Users/ramancini/src/python/ML-in-Remote-Sensing/ml/materials/jellybeans.tiff"
    jellybean_img = plt.imread(jellybean_path)

    # Reshape images into feature vectors
    jellybean_feat = jellybean_img.reshape([-1, 3])

    # Perform kmeans
    classified_jellybeans = kmeans(jellybean_feat, 6, 50)
    print(classified_jellybeans.dtype)

    # Display result
    fig, axs = plt.subplots(1, 2, sharey=True)
    axs[0].imshow(jellybean_img)
    axs[0].set_title("Jellybeans image")

    axs[1].imshow(classified_jellybeans.reshape([jellybean_img.shape[0], jellybean_img.shape[1]]))
    axs[1].set_title("K-Means clustered Jellybeans image")
    plt.show()

    # Load in Sentinel data
    path = "/Users/ramancini/src/python/ML-in-Remote-Sensing/eda/sentinel2_rochester.npy"
    bands = np.load(path)

    # Reshape the bands to match PCA function specs
    bands_reshape = bands.reshape([-1, bands.shape[2]])

    # Perform pca
    sent_pcs, sent_evals, sent_mean_arr = principal_component_analysis(bands_reshape)

    # Transform to lower dimensions (3, 4, 5, 6) and display
    comps = [3, 4, 5, 6]
    sent_fig, sent_axs = plt.subplots(2, 2, sharey=True)
    for idx, comp_idx in enumerate(comps):
        # Seed the numpy random number generator so labels end up similar
        np.random.seed(42)

        # Reproject the data
        reproj_data = np.dot(sent_mean_arr, sent_pcs[:,:comp_idx])

        # Perform kmeans and reshape data
        # Last HW = found there were roads, vegetation, water, try to replicate
        sent_classes = kmeans(reproj_data, k=4, max_iter=100)
        sent_classes_reshape = sent_classes.reshape([bands.shape[0], bands.shape[1]])

        # Display results
        ridx = idx // 2
        cidx = idx % 2

        img = sent_axs[ridx, cidx].imshow(sent_classes_reshape)
        sent_axs[ridx, cidx].set_title(
            f"Sentinel data kmeans with first {comp_idx} PCs")
        sent_fig.colorbar(img, ax=sent_axs[ridx, cidx], shrink=0.5)
    plt.show()

    # Load in hyperspectral data
    tait_data_path = '/Users/ramancini/src/python/ML-in-Remote-Sensing/ml/materials/tait_hsi'
    packed_data = spectral.envi.open(tait_data_path + '.hdr',
                                     tait_data_path)
    data = packed_data.load()

    # 375, 337
    # Grab interesting patch from data
    data_patch = data[41:291, 293:543, :]

    plt.imshow(data_patch[:,:,1],cmap='gray')
    plt.show()

    # Reshape the data for kmeans
    data_patch_reshape = data_patch.reshape([(data_patch.shape[0] * data_patch.shape[1]), data_patch.shape[2]])

    # Perform PCA
    tait_pcs, tait_evals, tait_mean_arr = principal_component_analysis(
        data_patch_reshape)

    # Perform kmeans
    tait_comps = [2, 5, 10, 50, 100]
    tait_results = []

    for comp_idx in tait_comps:
        # Reproject & reconstruct the data
        reproj_data = np.dot(tait_mean_arr, tait_pcs[:,:comp_idx])

        # Perform kmeans
        tait_labels = MiniBatchKMeans(n_clusters=6, max_iter=100).fit_predict(reproj_data)

        # Reshape
        tait_labels_reshape = tait_labels.reshape([data_patch.shape[0], data_patch.shape[1]])

        # Add to results list
        tait_results.append(tait_labels_reshape)

    # Perform kmeans on original data
    tait_labels = MiniBatchKMeans(n_clusters=6, max_iter=100).fit_predict(data_patch_reshape)

    # Reshape
    tait_labels_reshape = tait_labels.reshape(
        [data_patch.shape[0], data_patch.shape[1]])

    # Append to end of list
    tait_results.append(tait_labels_reshape)
    tait_comps.append('All')

    # Show results
    tait_fig, tait_axs = plt.subplots(2, 3, sharey=True)
    for idx in range(len(tait_results)):
        ridx = idx // 3
        cidx = idx % 3

        img = tait_axs[ridx, cidx].imshow(tait_results[idx])
        tait_axs[ridx, cidx].set_title(f"K-Means clustering with {tait_comps[idx]} PCs")
        tait_fig.colorbar(img, ax=tait_axs[ridx, cidx], shrink=0.5)

    plt.show()