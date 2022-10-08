#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .page import hOCR
from .clustering import KMeans, ClusterResults, silhouette_score
import numpy as np
import scipy.stats
from scipy.stats._stats_mstats_common import LinregressResult
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

class CoordData:

    def __init__(self, hocr: hOCR):
        """Extract the bounding boxes from a hOCR page."""
        arr = np.array(hocr.bboxes)
        # Store in a dictionary, along with the indicies
        self.coords = {
            'x0': arr[:,0],
            'y0': arr[:,1],
            'x1': arr[:,2],
            'y1': arr[:,3],
            'indices': np.array(range(len(arr)))
        }

    def fit_lm(self, x: str='', y: str=''):
        """Fit a least-squares linear regression model to the coordinates.

        Parameters
        ----------
        x
            Dimension 1 (e.g. x0, y0, etc.)
        y
            Dimension 2 (e.g. x1, y1, etc.)

        Raises
        ------
            ValueError if the coordinates are not in the data
        """
        if x not in self.coords or y not in self.coords:
            problem = ', '.join(list(self.coords.keys()))
            raise ValueError(f"Valid coordinate data: {problem}")

        x, y = self.coords[x], self.coords[y]
        # Fit the model
        self.lm = scipy.stats.linregress(x, y)
        # Make a vector coordinates to represent the regression line
        self.regression_line = self.lm.slope * x + self.lm.intercept

    def cluster(
        self,
        k: int,
        dim1: str='indices',
        dim2: str='x0',
        tol: float=0.001,
        iters: int=500
    ):
        """Do k-means clustering on the points.

        Parameters
        ----------
        k
            Number of clusters
        dim1
            First dimension
        dim2
            Second dimension
        tol
            The tolerance at which to stop clustering
        iters
            Number of iterations at which to stop clustering

        Raises
        ------
            ValueError if the specified dimensions are not in the data
        """
        if dim1 not in self.coords or dim2 not in self.coords:
            problem = ', '.join(list(self.coords.keys()))
            raise ValueError(f"Valid dimensions: {problem}")

        # Get the data to cluster on
        to_cluster = list(zip(self.coords[dim1], self.coords[dim2]))
        self.cluster_on = np.array(to_cluster)

        # Initialize and fit a k-means clusterer
        kmeans = KMeans(k, tol=tol, iters=iters)
        self.clusters = kmeans.predict(self.cluster_on)
        self.labels = kmeans.labels

def lm_model(hocr, x='indices', y='x0') -> LinregressResult:
    """Return a least-squares linear regression model.

    Parameters
    ----------
    x
        Dimension 1 (e.g. x0, y0, etc.)
    y
        Dimension 2 (e.g. x1, y1, etc.)

    Returns
    -------
    lm
        The results of the regression model

    """
    data = CoordData(hocr)
    data.fit_lm(x, y)

    return data.lm

def predict_columns(
    hocr: hOCR,
    max_k: int=4,
    dim1: str='indices',
    dim2: str='x0',
    r2: float=0.1,
    tol: float=0.001,
    iters: int=500,
) -> ClusterResults:
    """Use k-means clustering/silhouette scores to determine the number of
    columns.

    The idea here is that the optimal number of clusters in the dataset will
    correspond to the number of columns on the original page

    Parameters
    ----------
    hocr
        hOCR data
    max_k
        Maximum number of columns to test
    dim1
        First dimension
    dim2
        Second dimension
    r2
        A cutoff for the R^2 value; a value below this defaults to one column
    tol
        The tolerance at which to stop clustering
    iters
        The number of iterations at which to stop clustering

    Returns
    -------
    results
        A ClusterReesults object, which contains information about the results,
        best k, and labels
    """
    data = CoordData(hocr)

    # Before we do any predictions, we fit a linear model to the data to find
    # its R squared value (the proportion of variation in the data based on the
    # predicted variable of a linear model). If that value is below the `r2`
    # cutoff, the page is considered to be a single column
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

    # Get the optimal number of clusters, recluster on that, and get the label
    # assignments
    best_k = max(results, key=results.get)
    data.cluster(best_k, dim1=dim1, dim2=dim2, tol=tol, iters=iters)

    # Store everything in a container and return it
    return ClusterResults(results, best_k, data.labels)

def bbox_plot(
    hocr: hOCR,
    x: str='indices',
    y: str='x0',
    regression_line: bool=True,
    figx: int=12,
    figy: int=9,
    title: str=''
) -> Figure:
    """Plot coordinate data from the bounding boxes.

    Parameters
    ----------
    hocr
        hOCR data
    x
        Data for the X axis
    y
        Data for the Y axis
    regression_line
        Whether to fit a linear model and plot the regression line
    figx
        Figure width
    figy
        Figure height
    title
        Figure title

    Returns
    -------
    plot
        Linear plot
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
    hocr: hOCR,
    k: int,
    dim1: str='indices',
    dim2: str='x0',
    tol: float=0.001,
    iters: int=500,
    figx: int=12,
    figy: int=9,
    title: str=''
) -> Figure:
    """Do k-means clustering and plot it.

    Parameters
    ----------
    hocr
        hOCR data
    dim1
        First dimension
    dim2
        Second dimension
    tol
        The tolerance at which to stop clustering
    iters
        The number of iterations at which to stop clustering
    figx
        Figure width
    figy
        Figure height
    title
        Figure title

    Returns
    -------
    plot
        Scatter plot of clusters
    """
    data = CoordData(hocr)
    data.cluster(k, dim1=dim1, dim2=dim2, tol=tol, iters=iters)

    fig, ax = plt.subplots(figsize=(figx, figy))
    plt.scatter(data.coords[dim1], data.coords[dim2], c=data.labels)
    plt.title(title)
    plt.xlabel(dim1)
    plt.ylabel(dim2)

    plt.show()

