#!/usr/bin/env python3
"""10-kmeans.py"""
import sklearn.cluster


def kmeans(X, k):
    '''Performs K-means on a dataset.
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer, number of clusters
    Returns:
        C, clss or None, None on failure
        C: numpy.ndarray of shape (k, d) containing the centroid means for
        each cluster'''
    kmeans = sklearn.cluster.KMeans(n_clusters=k)
    kmeans.fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_
    return C, clss
