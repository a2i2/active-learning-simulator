#Basic imports
import os
import pprint
import ssl
from datetime import datetime
import time

import numpy as np
import pandas as pd

#sklearn imports
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA #Principal Component Analysis
from sklearn.manifold import TSNE #T-Distributed Stochastic Neighbor Embedding
from sklearn.cluster import KMeans #K-Means Clustering
from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'

#plotly imports
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


from celluloid import Camera


from active_learner import ActiveLearner
from command_line_interface import parse_CLI
from data_extraction import get_datasets


working_directory = './'


def main():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # set up output directory
    output_name = str(datetime.now())
    output_directory = working_directory + output_name
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    # get desired parameters for training
    params = parse_CLI()
    pp = pprint.PrettyPrinter()
    param = params[0]
    print()
    pp.pprint(param)
    print()

    datasets = get_datasets(param['data'][0], param['data'][1], working_directory, param['data'][2])

    n = 3
    for i in range(len(datasets)):
        output_path = output_directory + "/cluster_" + str(i) + ".mp4"
        visualise_clustering(n, datasets[i], param, output_path)


def visualise_clustering(n, data, param, output_path):
    AL = run_model(data, param)
    print("Active learning complete")
    print("Recall:", AL.evaluator.recall[-1])
    cluster_method = KMeans(n_clusters=n)
    cluster_eval = ClusterEvaluation(cluster_method, n)
    cluster_eval.train(data)
    cluster_eval.plot_progress(*cluster_eval.make_figure(), AL, output_path)


def run_model(data, params):
    """
    Creates algorithm objects and runs the active learning program
    :param data: dataset for systematic review labelling
    :param params: input parameters for algorithms and other options for training
    :return: returns the evaluator and stopper objects trained on the dataset
    """
    N = len(data)
    # determine suitable batch size, batch size increases with increases dataset size
    batch_size = int(0.03 * N)
    # TODO different batch sizes? as a parameter

    # create algorithm objects
    model_AL = params['model'][0](*params['model'][1])
    selector = params['selector'][0](batch_size, params['confidence'], *params['selector'][1], verbose=params['selector'][2])
    stopper = params['stopper'][0](N, params['confidence'], *params['stopper'][1], verbose=params['stopper'][2])

    # specify evaluator object if desired
    evaluator = None
    if params['evaluator']:
        evaluator = params['evaluator'][0](data, verbose=params['evaluator'][1])

    # create active learner
    active_learner = ActiveLearner(model_AL, selector, stopper, batch_size=batch_size, max_iter=1000,
                                   evaluator=evaluator, verbose=params['active_learner'][1])

    # train active learner
    (mask, relevant_mask) = active_learner.train(data)
    return active_learner


def indices_to_mask(indices, N, full_relevant_mask):
    indice_mask = np.zeros(N, dtype=np.uint8)
    indice_mask[indices] = 1

    relevant_mask = np.zeros(N, dtype=np.uint8)
    relevant_mask[indices] = full_relevant_mask[indices]
    return indice_mask, relevant_mask


def get_cluster_colour(n, N, X):
    colours = np.zeros((N, 4))
    edge_colours = np.zeros((N, 4))
    for i in range(n):
        # change colouring for each cluster
        red = i * 200.0 / (n - 1)
        green = (n - 1 - i) * 180.0 / (n - 1)
        blue = 250
        alpha = 0.8
        colour = [red / 300, green / 300, blue / 300, alpha]
        indices = X["cluster"] == i
        colours[indices] = colour
        edge_colours[indices] = [red / 300, green / 300, blue / 300, 0]
    return colours, edge_colours


class ClusterEvaluation:

    def __init__(self, clustering_method, n):
        self.progress = None
        self.clustering_method = clustering_method
        self.clusters = None
        self.n_clusters = n
        self.dimension = 2
        self.Y = None
        self.X = None
        self.N = None

    def train(self, data):
        self.N = len(data)
        self.X = data['x'].copy(deep=True)
        self.Y = data['y'].copy(deep=True)
        self.X = self.fit(self.X)
        self.X = self.predict(self.X['x_scaled'])
        self.X = self.compute_PCA(self.X)
        #self.plot(*self.make_figure(), AL.indice_mask, AL.relevant_mask)

    def fit(self, X):
        X['x_scaled'] = self.scale_features(X)
        self.clustering_method.fit(X['x_scaled'])
        return X

    def scale_features(self, data):
        X = data.to_list()
        scaler = StandardScaler()
        print("1")
        scaled_features = pd.DataFrame(scaler.fit_transform(X))
        print("2")
        return scaled_features

    def predict(self, X):
        self.clusters = self.clustering_method.predict(X)
        X['cluster'] = self.clusters
        return X

    def compute_PCA(self, X):
        pca_2D = PCA(n_components=self.dimension)
        PCs_2D = pd.DataFrame(pca_2D.fit_transform(X))
        column_names = []
        for i in range(self.dimension):
            name = "PC{axis}_{dimension}D".format(axis=i+1, dimension=self.dimension)
            column_names.append(name)
        PCs_2D.columns = column_names
        new_X = pd.concat([X, PCs_2D], axis=1, join='inner')
        return new_X

    def make_figure(self):
        fig = plt.figure(constrained_layout=True, dpi=600)
        ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
        ax.set(aspect=1)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_facecolor((0.05, 0, 0.1))
        return fig, ax

    def plot(self, fig, ax, indice_mask, relevant_mask):
        # set up base cluster colours
        colours, edge_colours = get_cluster_colour(self.n_clusters, self.N, self.X)
        # un-screened and irrelevant
        indices = (indice_mask == 0) & (relevant_mask == 0)
        colours[indices, 0:3] *= 0.3
        ax.scatter(self.X[indices]["PC1_2D"], self.X[indices]["PC2_2D"], s=2, c=colours[indices], linewidths=0)
        # screened and irrelevant
        indices = (indice_mask == 1) & (relevant_mask == 0)
        ax.scatter(self.X[indices]["PC1_2D"], self.X[indices]["PC2_2D"], s=2, c=colours[indices], linewidths=0)
        # un-screened and relevant
        indices = (indice_mask == 0) & (relevant_mask == 1)
        edge_colours[indices] = [1, 0, 0, 1]
        ax.scatter(self.X[indices]["PC1_2D"], self.X[indices]["PC2_2D"], s=5, c=colours[indices], edgecolors=edge_colours[indices], linewidths=0.2)
        # screened and relevant
        indices = (indice_mask == 1) & (relevant_mask == 1)
        edge_colours[indices] = [1, 1, 1, 1]
        ax.scatter(self.X[indices]["PC1_2D"], self.X[indices]["PC2_2D"], s=5, c=colours[indices], edgecolors=edge_colours[indices], linewidths=0.2)
        return fig, ax

    def plot_progress(self, fig, ax, AL, output_directory, verbose=False):
        if verbose:
            def progress(p):
                print('Progress::', str(p) + "/" + str(self.N))
            self.progress = progress
        else:
            self.progress = lambda *a: None

        batch_size = AL.batch_size // 10
        camera = Camera(fig)
        for i in range(0, len(AL.evaluator.screen_indices) - 1, batch_size):
            indices = AL.evaluator.screen_indices[0: i + 1]
            indice_mask, relevant_mask = indices_to_mask(indices, AL.N, AL.relevant_mask)
            fig, ax = self.plot(fig, ax, indice_mask, relevant_mask)
            camera.snap()
            self.progress(i)
        indices = AL.evaluator.screen_indices
        indice_mask, relevant_mask = indices_to_mask(indices, AL.N, AL.relevant_mask)
        fig, ax = self.plot(fig, ax, indice_mask, relevant_mask)
        camera.snap()
        self.progress(self.N)

        anim = camera.animate(blit=True)
        print("animation created")
        anim.save(output_directory, fps=30, dpi=600)
        print("animation saved")
        print()
        return


if __name__ == '__main__':
    main()
