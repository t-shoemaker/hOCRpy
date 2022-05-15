#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

def bbox_plot(hocr, x='indices', y='x0', regression_line=True, figx=12, figy=9, title=''):
    """Plot coordinate data from the bounding boxes."""
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

def lm_model(hocr, x='indices', y='x0'):
    """Return a least-squares linear regression model."""
    data = CoordData(hocr)
    data.fit_lm(x, y)

    return data.lm
