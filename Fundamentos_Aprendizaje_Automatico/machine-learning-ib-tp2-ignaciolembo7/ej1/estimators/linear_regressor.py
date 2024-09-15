from .base import SGDEstimator
import numpy as np

class LinearRegressor(SGDEstimator):
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.coef_) + self.intercept_

    def _gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_pred = np.dot(X, self.coef_) + self.intercept_
        gradient = np.dot(X.T, y_pred - y) / len(X)
        gradient_intercept = np.mean(y_pred - y) 
        return gradient_intercept, gradient
