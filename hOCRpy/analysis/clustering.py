#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import cdist

class KMeans:

    def __init__(self, k, tol=0.001, iters=500):
        """Predefine number of clusters as well as the tolerance and iterations.

        :param k: The number of clusters
        :type k: int
        :param tol: Cutoff for stopping the clustering
        :type tol: float
        :param iters: Number of iterations for another cutoff method
        :type iters: int
        """
        if k < 2:
            raise ValueError("You must select more than one cluster!")

        self.k = k
        self.tol = tol
        self.iters = iters

    def _dist(self, a, b):
        """Compute the Euclidean distance between matrix a and matrix b.

        :param a: First matrix
        :type a: np.array
        :param b: Second matrix
        :type b: np.array
        :returns: A vector of distances
        :rtype: np.array
        """
        # Get the square of each matrix
        a_square = np.reshape(np.sum(a * a, axis=1), (a.shape[0], 1))
        b_square = np.reshape(np.sum(b * b, axis=1), (1, b.shape[0]))

        # Multiply the two matrices, taking the transpose of the second
        ab = a @ b.T

        return np.sqrt(-2 * ab + b_square + a_square)

    def _get_clusters(self, centroids, return_labels=False):
        """Determine which observations are closest to each cluster.

        :param centroids: The center points
        :type centroids: np.array
        :param return_labels: Return the cluster labels for each data point
        :type return_labels: bool
        :returns: If not return_labels, a dictionary of cluster: data pairs; otherwise, labels
        :rtype: dict, np.array
        """
        # Create a dict to hold the clusters
        clusters = {}
        for i in range(self.k):
            clusters[i] = []

        # Find the distance between each observation and the centroids
        distances = self._dist(self.data, centroids)

        # For each observation, find the index of the minimum value. This 
        # index will correspond to the cluster
        closest = np.argmin(distances, axis=1)
        for idx, cluster_id in enumerate(closest):
            clusters[cluster_id].append(self.data[idx])

        # If we just want the labels, we send back the `closest` array
        if return_labels:
            return closest

        return clusters

    def _converged(self, old_centroids, new_centroids):
        """Check to see whether the centroids are still moving.

        This is a simple check: look at the distance between the old centroids
        and the new ones. If it hasn't changed much, we're done. The tolerance 
        metric defines the cutoff

        :param old_centroids: Prior centers of clusters
        :type old_centroids: np.array
        :param new_centroids: New centers of clusters
        :type new_centroids: np.array
        :returns: Indicate whether the centers should continue moving
        :rtype: bool
        """
        # Get distances
        distances = self._dist(old_centroids, new_centroids)

        # Are the distances under the tolerance? We check the diagonal of the 
        # distance matrix we've just created
        converged = np.max(distances.diagonal()) <= self.tol

        return converged

    def predict(self, data):
        """Fit the clusterer and partition the data into k clusters."""
        self.data = data
        n_samples = data.shape[0]

        # Randomly pick k samples from the data as the seed centroids
        seed_idx = np.random.choice(range(0, n_samples), self.k)
        seed = np.array([self.data[idx] for idx in seed_idx])

        # Begin by assuming the centroids exceed our tolerance
        converged = False

        # Depending on the points chosen, it can take a while for a clusterer to 
        # Converge. We optionally include a max iteration cutoff
        iteration = 0
        while not converged and iteration < self.iters:
            old_centroids = seed.copy()
            clusters = self._get_clusters(old_centroids)
            # Generate new centroids by getting the mean of the clusters we've 
            # just created
            seed = np.array([
                np.mean(clusters[key], axis=0, dtype=self.data.dtype)
                for key in sorted(clusters.keys())
            ])
            # Check whether the clusters have converged
            converged = self._converged(old_centroids, seed)
            iteration += 1

        self.converged = converged
        self.fit_iters = iteration
        self.centroids = seed.copy()
        self.clusters = self._get_clusters(seed)
        self.labels = self._get_clusters(seed, return_labels=True)

        return self.clusters

class ClusterResults:

    def __init__(self, results, best_k, labels):
        """Initialize a container to hold information about a clustering run/analysis.

        :param results: The silhouette scores for a run of k clusters
        :type results: dict
        :param best_k: An optimal number of clusters
        :type best_k: int
        :param labels: Label assignments for each token in the hOCR
        :type labels: np.array
        """
        self.results = results
        self.best_k = best_k
        self.labels = labels
       
    def __repr__(self):
        output = f"Column prediction\n-----------------\n"
        output += f"Silhouette Scores: {self.results}\nBest k: {self.best_k}"
        return output
 
def silhouette_score(data, labels):
    """Calculate the silhouette score for k clusters on a dataset.

    :param data: The dataset
    :type data: np.array
    :param labels: Cluster assignments for each point in the dataset
    :type labels: np.array
    :returns: Silhouette score
    :rtype: int
    """
    # Partition the data into their respective clusters
    clusters = {k: np.where(labels == k) for k in np.unique(labels)}

    # Now, roll through each cluster
    scores = []
    for k in clusters:
        # Select the relevant data points
        cluster = data[clusters[k]]
        # Get the mean Euclidean distance between all points in this cluster
        mean_dist = np.mean(cdist(cluster, cluster, 'euclidean'))

        # Get the mean Euclidean distance between all points in this cluster 
        # and every other cluster
        other_k_dist = [
            np.mean(cdist(cluster, data[clusters[_k]])) for _k in clusters
            if _k != k
        ]

        # Find the closest other cluster (it will be the shortest distance
        other_k_dist = np.min(other_k_dist)

        # Finding the silhouette score is relatively straightforward: take the 
        # other cluster's distance and subtract it from this cluster's mean
        # distance. Then divide that by the max of both values
        coeff = (other_k_dist - mean_dist) / max(mean_dist, other_k_dist)
        scores.append(coeff)

    # Return the average silhouette score for all clusters
    return np.mean(scores)
