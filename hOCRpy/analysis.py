#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .page import hOCR
from .clustering import KMeans, ClusterResults, silhouette_score
from .linregress import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

class CoordData:

    def __init__(self, hocr: hOCR):
        """Extract the bounding boxes from a hOCR page."""
        arr = np.array(hocr.bboxes)
        # Store in a dictionary, along with the indicies
        self.data = {
            'x0': arr[:,0],
            'y0': arr[:,1],
            'x1': arr[:,2],
            'y1': arr[:,3],
            'indices': np.array(range(len(arr)))
        }

    def fit_lm(
        self, x: str='',
        y: str='',
        test_size: float=.2,
        alpha: float=0.5,
        tol: float=1e-5,
        iters: int=500
    ) -> None:
        """Fit a least-squares linear regression model to the coordinates.

        Parameters
        ----------
        x
            Dimension 1 (e.g. x0, y0, etc.)
        y
            Dimension 2 (e.g. x1, y1, etc.)
        test_size
            The proportional size of the testing data
        alpha
            Learning rate
        tol
            The tolerance at which to stop training
        iters
            The iterations at which to stop training

        Raises
        ------
            ValueError if the coordinates are not in the data
        """
        if x not in self.data or y not in self.data:
            valid = ', '.join(list(self.data.keys()))
            raise ValueError(f"Valid coordinate data: {valid}")

        x, y = self.data[x], self.data[y]
        model = LinearRegression(x, y)
        model.train(test_size, alpha, tol, iters)
        self.lm = model

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
        if dim1 not in self.data or dim2 not in self.data:
            validm = ', '.join(list(self.data.keys()))
            raise ValueError(f"Valid dimensions: {valid}")

        # Get the data to cluster on
        to_cluster = list(zip(self.data[dim1], self.data[dim2]))
        self.cluster_on = np.array(to_cluster)

        # Initialize and fit a k-means clusterer
        kmeans = KMeans(k, tol=tol, iters=iters)
        self.clusters = kmeans.predict(self.cluster_on)
        self.labels = kmeans.labels

def lm_model(
    hocr: hOCR,
    x: str='indices',
    y: str='x0',
    test_size: float=.2,
    alpha: float=0.5,
    tol: float=1e-5,
    iters: int=500
) -> LinearRegression:
    """Return a least-squares linear regression model.

    Parameters
    ----------
    x
        Dimension 1 (e.g. x0, y0, etc.)
    y
        Dimension 2 (e.g. x1, y1, etc.)
    test_size
        The proportional size of the testing data
    alpha
        Learning rate
    tol
        The tolerance at which to stop training
    iters
        The iterations at which to stop training

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
    # its R-Squared value (the proportion of variation in the data based on the
    # predicted variable of a linear model). If that value is below the `r2`
    # cutoff, the page is considered to be a single column
    data.fit_lm(dim1, dim2)
    if data.lm.rsquared < r2:
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
    test_size: float=.2,
    alpha: float=0.5,
    tol: float=1e-5,
    iters: int=500,
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
    test_size
        The proportional size of the testing data
    alpha
        Learning rate
    tol
        The tolerance at which to stop training
    iters
        The number of iteratinos at which to stop training
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
    x_data, y_data = data.data[x], data.data[y]

    fig, ax = plt.subplots(figsize=(figx, figy))
    plt.plot(x_data, y_data)

    # Optionally add a regression line
    if regression_line:
        data.fit_lm(x, y, test_size, alpha, tol, iters)
        plt.plot(x_data, data.lm.regression_line, c='red', label="Regression line")
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
    coords = CoordData(hocr)
    coords.cluster(k, dim1=dim1, dim2=dim2, tol=tol, iters=iters)

    fig, ax = plt.subplots(figsize=(figx, figy))
    plt.scatter(coords.data[dim1], coords.data[dim2], c=coords.labels)
    plt.title(title)
    plt.xlabel(dim1)
    plt.ylabel(dim2)

    plt.show()

