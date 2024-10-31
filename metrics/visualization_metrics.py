import numpy as np
import sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt



#Define function for dimensionaltiy reduction, transforming high-dimensional data into low-dimensional representation
def low_dimensional_representation(data_ori: np.ndarray, data_gen: np.ndarray, technique: str) -> np.ndarray:
    """Transform data into low-dimensional representation."""
    assert technique in ['pca', 'tsne']

    sample_no = min([1000, len(data_ori)])
    idx = np.random.permutation(len(data_ori))[:sample_no]
    data_ori, data_gen = np.asarray(data_ori)[idx], np.asarray(data_gen)[idx]
    no, seq_len, dim = data_ori.shape

    prep_data = np.array([np.reshape(np.mean(data_ori[i,:,:], 1), [1,seq_len]).flatten().tolist() for i in range(sample_no)])
    prep_data_hat = np.array([np.reshape(np.mean(data_gen[i,:,:], 1), [1,seq_len]).flatten().tolist() for i in range(sample_no)])

    if technique == 'pca':
        # PCA Analysis
        pca = PCA(n_components = 2)
        pca.fit(prep_data)

        return pca.transform(prep_data), pca.transform(prep_data_hat)

    elif technique == 'tsne':
        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)

        # TSNE anlaysis
        tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, max_iter = 300)
        tsne_result = tsne.fit_transform(prep_data_final)

        return tsne_result[:sample_no], tsne_result[sample_no:]



#Plotting function
def plot_distribution_estimate(lowdim_data_ori: np.ndarray, lowdim_data_gen: np.ndarray, technique: str) -> None:
    """Visualize distribution estimate via sampled data points."""
    fig, ax = plt.subplots()
    #Use rasterized Line2D Artist here instead of scatter. Espacially benefitial when exporting the plot as pdf file (or other vector graphics).
    ax.add_line(mpl.lines.Line2D(lowdim_data_ori[:,0], lowdim_data_ori[:,1], ls='', marker='o', markersize=3, c = 'red', alpha = 0.2, rasterized=True, label='Original'))
    ax.add_line(mpl.lines.Line2D(lowdim_data_gen[:,0], lowdim_data_gen[:,1], ls='', marker='o', markersize=3, c = 'blue', alpha = 0.2, rasterized=True, label='Generated'))
    ax.autoscale_view()
    ax.set_xlabel(f'x-{technique}')
    ax.set_ylabel(f'y-{technique}')
    ax.legend()
    fig.show()