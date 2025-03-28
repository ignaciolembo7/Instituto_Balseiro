from .base import SGDEstimator
import numpy as np

class LogisticClassifier(SGDEstimator):
    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = np.dot(X, self.coef_) + self.intercept_
        return 1 / (1 + np.exp(-logits))

    def _gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        logits = np.dot(X, self.coef_) + self.intercept_
        y_pred = 1 / (1 + np.exp(-logits))
        gradient = np.dot(X.T, y_pred - y) / len(X)
        gradient_intercept = 2*np.mean(y_pred - y) 
        return gradient_intercept, gradient 