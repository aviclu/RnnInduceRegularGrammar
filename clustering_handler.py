import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth


def get_best_kmeans_model(X, max_k):
    """
    dividing the states into k clusters, using k-means algorithm.
    there are several methods to find the best k:
    https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set
    we will use the x-means clustering: keep looking for the best k until BIC is reached.

    maximize BIC(C | X) = L(X | C) - (p / 2) * log n, where:
    C - model, X - dataset, p - number of parameters in C, n - number of points in the dataset

    :param X: input - in our case, all possible states
    :return: the best k-means model, by BIC - which has n clusters, such that n << size(states)
    """
    # todo: maybe we should use this code instead of implementing it ourselves:
    # https://github.com/mynameisfiber/pyxmeans/blob/master/pyxmeans/xmeans.py

    kmeans_models = [KMeans(n_clusters=k).fit(X) for k in range(1, max_k)]
    best_bic = np.argmax([compute_bic(model, X) for model in kmeans_models])
    return kmeans_models[best_bic]


def compute_bic(model, X):
    """
    Computes the BIC metric for a given model
    :param model: k-means model object
    :param X: the input, multidimensional np array of data points
    :return: BIC value
    """
    centers = [model.cluster_centers_]
    labels = model.labels_
    m = model.n_clusters  # number of clusters
    n = np.bincount(labels)  # size of the clusters
    N, d = X.shape  # size of data set

    # compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], np.array([centers[0][i]]),
                                                           'euclidean') ** 2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d + 1)

    bic_value = np.sum([n[i] * np.log(n[i]) -
                        n[i] * np.log(N) -
                        ((n[i] * d) / 2) * np.log(2 * np.pi * cl_var) -
                        ((n[i] - 1) * d / 2) for i in range(m)]) - const_term

    return bic_value


def get_best_meanshift_model(X):
    # TODO quantile hyperparameter can be changed
    bandwidth = estimate_bandwidth(X, quantile=0.3, n_samples=X.shape[0])
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    return ms


def get_cluster(state_vector, model):
    center_idx = model.predict([state_vector])[0]
    return model.cluster_centers_[center_idx]
