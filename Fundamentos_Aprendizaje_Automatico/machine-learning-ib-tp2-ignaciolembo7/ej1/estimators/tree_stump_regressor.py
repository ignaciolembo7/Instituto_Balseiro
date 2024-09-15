import numpy as np
from .base import BaseEstimator

class TreeStumpRegressor(BaseEstimator):
    """
    A simple decision stump that uses a single feature and threshold to make predictions.
    The stump chooses the best feature and threshold based on minimizing the mean squared error.
    """
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_mean = None
        self.right_mean = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        best_error = np.inf

        for feature_idx in range(n_features):

            X_sorted = np.sort(X[:,feature_idx])
            thresholds = (X_sorted[1:] + X_sorted[:-1]) / 2

            for threshold in thresholds:  #np.unique(X[:,feature_idx]):
                left_values = y[X[:,feature_idx] < threshold]
                left_mean = left_values.mean()  
                right_values = y[X[:,feature_idx] >= threshold]
                right_mean = right_values.mean()

                error = np.sum((left_values - left_mean)**2)  + np.sum((right_values - right_mean)**2) 

                if error < best_error:
                    best_error = error
                    self.feature_index = feature_idx
                    self.threshold = threshold
                    self.left_mean = left_mean
                    self.right_mean = right_mean

    def predict(self, X: np.ndarray) -> np.ndarray:
        feature_values = X[:,self.feature_index]
        return np.where(feature_values >= self.threshold, self.right_mean, self.left_mean)
