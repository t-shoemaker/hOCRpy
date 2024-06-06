#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple


class LinearRegression:
    def __init__(self, x: np.array, y: np.array):
        """Initialize by binding X and Y data.

        Adapted from
        https://www.cs.toronto.edu/~frossard/post/linear_regression/
        """
        # Ensure the data are floats
        if x.dtype != float:
            x = x.astype("float64")
            y = y.astype("float64")

        # Load in the X values and normalize them
        self.x = x[:, np.newaxis]
        self.x /= np.max(x)
        self.x = np.hstack((np.ones_like(self.x), self.x))
        self.y = y
        self.num_obvs = len(x)

        # The following will be modified during training: the gradient values,
        # the coeffiecients (which we randomly populate for now), the cost
        # function, the R-squared value, the slope, and the regression line. We
        # also set a flag for whether the model is trained and whether it has
        # converged
        self.gradient = np.zeros(2)
        self.w = np.random.random(2)
        self.mse = 0.0
        self.test_mse = 0.0
        self.rsquared = 0.0
        self.slope = 0.0
        self.regression_line = np.zeros(self.num_obvs)

        self.trained = False
        self.converged = False

    def __repr__(self):
        output = "Regression model\n"
        output += f"+ Trained: {self.trained}\n"
        output += f"+ Converged: {self.converged}\n"
        output += f"+ Test error: {self.test_mse:0.4f}\n"
        output += f"+ Slope: {self.slope:0.4f}\n"
        output += f"+ R-Squared: {self.rsquared:0.4f}"

        return output

    @staticmethod
    def _cost(w: np.array, x: np.array, y: np.array) -> Tuple[np.array, float]:
        """Calculate gradient descent.

        Parameters
        ----------
        w
            Coefficients
        x
            Dimension 1
        y
            Dimension 2

        Returns
        -------
        gradient
            Gradient
        mse
            Mean-squared error
        """
        y_estimate = (x @ w).flatten()
        error = y.flatten() - y_estimate
        mse = (1.0 / len(x)) * np.sum(np.power(error, 2))
        gradient = -(1.0 / len(x)) * (error @ x)

        return gradient, mse

    @staticmethod
    def _rsquared(pred: np.array, actual: np.array) -> float:
        """Calculate the R-squared value for a regression.

        Predictions
        -----------
        pred
            Predicted points along the regression line
        actual
            Actual points

        Returns
        -------
        r
            The R-squared value
        """
        r = np.sum((pred - actual.mean()) ** 2) / np.sum(
            (actual - actual.mean()) ** 2
        )

        return r

    def _split_data(self, test_size: float = 0.2) -> Tuple[np.array]:
        """Split data into train and test batches.

        Parameters
        ----------
        test_size
            The proportional size of the testing data

        Returns
        -------
        batched
            Data split into train/test batches

        Raises
        ------
        AssertionError
            If the test_size is >=1
        """
        # Ensure the test size is acceptable
        assert test_size <= 1, "`test_size` must be less than 1"

        # Shuffle the data and find the index at which to make the split
        order = np.random.permutation(self.num_obvs)
        portion = int(round(test_size * self.num_obvs))
        # Split
        test_x, train_x = self.x[order[:portion]], self.x[order[portion:]]
        test_y, train_y = self.y[order[:portion]], self.y[order[portion:]]

        return test_x, train_x, test_y, train_y

    def fit(
        self,
        test_size: float = 0.2,
        alpha: float = 0.5,
        tol: float = 1e-5,
        iters: int = 500,
        verbose: bool = False,
    ) -> None:
        """Train the model.

        Parameters
        ----------
        test_size
            The proportional size of the testing data
        alpha
            Learning rate
        tol
            The tolerance at which to stop training
        iters
            The iterations at which to stop training
        verbose
            Send training progress updates to screen
        """
        if verbose:
            print("Fitting a linear regression model...")

        # Split the data and create a counter
        test_x, train_x, test_y, train_y = self._split_data(test_size)
        iteration = 0

        while True:
            # Calculate the cost, then update the weights with the learning
            # rate and new gradient
            self.gradient, self.mse = self._cost(self.w, train_x, train_y)
            new_w = self.w - alpha * self.gradient

            # If we're below the tolerance, break
            if np.sum(abs(new_w - self.w)) < tol:
                self.converged = True
                break

            # If we're above the desired iterations, break
            if iteration > iters:
                break

            # Verbose training
            if verbose:
                if iteration % 100 == 0:
                    err = np.round(self.mse, 4)
                    print(
                        f"Iteration: {iteration:>4}",
                        f"| Mean-squared error: {err:>6}",
                    )

            iteration += 1
            self.w = new_w

        # Set the trained flag
        self.trained = True

        # Calculate the cost on the testing data
        _, self.test_mse = self._cost(self.w, test_x, test_y)

        # Set the regression line, its slope, and the R-squared value
        self.regression_line = self.x @ self.w
        self.slope = np.gradient(self.regression_line).mean()
        self.rsquared = self._rsquared(self.regression_line, self.y)
