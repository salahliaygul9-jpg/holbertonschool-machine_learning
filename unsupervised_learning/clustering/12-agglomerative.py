#!/usr/bin/env python3
"""Agglomerative clustering module"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Performs agglomerative clustering on a dataset."""
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')

    fig, ax = plt.subplots()
    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist, ax=ax)
    plt.show()

    clss = scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')

    return clss
