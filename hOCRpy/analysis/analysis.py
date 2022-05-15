#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .clustering import KMeans, silhouette_score
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

class CoordData:

    def __init__(self, hocr):
        """Extract the bounding boxes from a hOCR page."""
        x0 = np.array([b[0] for b in hocr.bboxes])
        y0 = np.array([b[1] for b in hocr.bboxes])
        x1 = np.array([b[2] for b in hocr.bboxes])
        y1 = np.array([b[3] for b in hocr.bboxes])
        # Store in a dictionary, along with the indicies
        self.coords = {
            'x0': x0,
            'y0': y0,
            'x1': x1,
            'y1': y1,
            'indices': np.array(range(len(hocr.bboxes)))
        }

    def fit_lm(self, x='', y=''):
        """Fit a least-squares linear regression model to the coordinates."""
        if x not in self.coords or y not in self.coords:
            raise ValueError(f"Valid coordinate data: {', '.join(list(self.coords.keys()))}")

        x, y = self.coords[x], self.coords[y]
        # Fit the model
        self.lm = scipy.stats.linregress(x, y)
        # Make a vector coordinates to represent the regression line
        self.regression_line = self.lm.slope * x + self.lm.intercept

    def cluster(self, k, dim1='indices', dim2='x0', tol=0.001, iters=500):
        """Do k-means clustering on the points.

        :param k: The number of clusters
        :type k: int
        :param dim1: First dimension
        :type dim1: str
        :param dim2: Second dimension
        :type dim2: str
        :param tol: The tolerance at which to stop clustering
        :type tol: float
        :param iters: The number of iterations at which to stop clustering
        :type iters: int
        """
        if dim1 not in self.coords or dim2 not in self.coords:
            raise ValueError(f"Valid dimensions: {', '.join(list(self.coords.keys()))}")

        # Get the data to cluster on
        to_cluster = list(zip(self.coords[dim1], self.coords[dim2]))
        self.cluster_on = np.array(to_cluster)

        # Initialize and fit a k-means clusterer
        kmeans = KMeans(k, tol=tol, iters=iters)
        self.clusters = kmeans.predict(self.cluster_on)
        self.labels = kmeans.labels

def lm_model(hocr, x='indices', y='x0'):
    """Return a least-squares linear regression model."""
    data = CoordData(hocr)
    data.fit_lm(x, y)

    return data.lm

def predict_columns(
    hocr,
    max_k=4,
    dim1='indices',
    dim2='x0',
    r2=0.1,
    tol=0.001,
    iters=500,
    return_scores=False
    ):
    """Use k-means clustering/silhouette scores to determine the number of columns.

    The idea here is that the optimal number of clusters in the dataset will correspond to 
    the number of columns on the original page

    :param hocr: hOCR data
    :type hocr: xml
    :param max_k: Maximum number of columns to test
    :type max_k: int
    :param dim1: First dimension
    :type dim1: str
    :param dim2: Second dimension
    :type dim2: str
    :param r2: A cutoff for the R squared value; a value below this defaults to one column
    :type r2: float
    :param tol: The tolerance at which to stop clustering
    :type tol: float
    :param iters: The number of iterations at which to stop clustering
    :type iters: int
    :param return_scores: Whether to return all of the silhouette scores
    :type return_scores: bool
    :returns: If `return_scores`, return clusters with their scores, otherwis return optimal k
    :type: dict, int
    """
    data = CoordData(hocr)

    # Before we do any predictions, we fit a linear model to the data to find its R squared 
    # value (the proportion of variation in the data based on the predicted variable of a linear 
    # model). If that value is below the `r2` cutoff, the page is considered to be a single 
    # column
    data.fit_lm(dim1, dim2)
    data_r2 = data.lm.rvalue ** 2
    if data_r2 < r2:
        return 1

    # Grid search the k space
    results = {}
    for k in range(2, max_k):
        data.cluster(k, dim1=dim1, dim2=dim2, tol=tol, iters=iters)
        score = silhouette_score(data.cluster_on, data.labels)
        results[k] = score

    # Return the cluster: score dictionary
    if return_scores:
        return results

    # Or return the optimal number of clusters
    return max(results, key=results.get)

def bbox_plot(
    hocr,
    x='indices',
    y='x0',
    regression_line=True,
    figx=12,
    figy=9,
    title=''
    ):
    """Plot coordinate data from the bounding boxes.

    :param hocr: hOCR data
    :type hocr: xml
    :param x: Data for x axis
    :type x: int
    :param y: Data for y axis
    :type y: int
    :param regression_line: Whether to fit a linear model and plot the regression line
    :type regression_line: bool
    :param figx: Figure width
    :type figx: int
    :param figy: Figure height
    :type figy: int
    :param title: Figure title
    :type title: str
    :returns: Linear plot
    :rtype: pyplot
    """
    data = CoordData(hocr)
    x_data, y_data = data.coords[x], data.coords[y]

    fig, ax = plt.subplots(figsize=(figx, figy))
    plt.plot(x_data, y_data)

    # Optionally add a regression line
    if regression_line:
        data.fit_lm(x, y)
        plt.plot(x_data, data.regression_line, c='red', label="Regression line")
        plt.legend()

    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)

    plt.show()

def cluster_plot(
    hocr,
    k,
    dim1='indices',
    dim2='x0',
    tol=0.001,
    iters=500,
    figx=12,
    figy=9,
    title=''
    ):
    """Do k-means clustering and plot it.

    :param hocr: hOCR data
    :type hocr: xml
    :param k: The number of clusters
    :type k: int
    :param dim1: First dimension
    :type dim1: str
    :param dim2: Second dimension
    :type dim2: str
    :param tol: The tolerance at which to stop clustering
    :type tol: float
    :param iters: The number of iterations at which to stop clustering
    :type iters: int
    :param figx: Figure width
    :type figx: int
    :param figy: Figure height
    :type figy: int
    :param title: Figure title
    :type title: str
    :returns: Cluster plot
    :rtype: pyplot
    """
    data = CoordData(hocr)
    data.cluster(k, dim1=dim1, dim2=dim2, tol=tol, iters=iters)

    fig, ax = plt.subplots(figsize=(figx, figy))
    plt.scatter(data.coords[dim1], data.coords[dim2], c=data.labels)
    plt.title(title)
    plt.xlabel(dim1)
    plt.ylabel(dim2)

    plt.show()

