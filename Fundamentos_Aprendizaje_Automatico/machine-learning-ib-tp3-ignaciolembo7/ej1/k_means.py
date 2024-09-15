import numpy as np

class KMeans:
    def __init__(self, k: int, seed: int):
        """
        Initialize KMeans clustering model.
        
        Args:
        - k (int): Number of clusters.
        - seed (int): Seed to randomly initialize the centroids.
        """
        self.k = k
        self.seed = seed
        self.centroids = None
    
    def fit(self, X: np.ndarray, n_iter: int=100) -> np.ndarray:
        """
        Perform clustering on input data X into k clusters.
        Args:
        - X (np.ndarray): Input array containing data to cluster.
        - n_iter (int): Number of iterations for the KMeans algorithm.
        Returns:
        - np.ndarray: Array containing cluster labels assigned to each point.
        """
        np.random.seed(self.seed)
        indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[indices]
        for _ in range(n_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            self._update_centroids(X, labels)
        return labels
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> None:
        """
        Update centroids of clusters based on point labels.

        Args:
        - X (np.ndarray): Input array containing data.
        - labels (np.ndarray): Array containing cluster labels assigned to each point.
        """
        for i in range(self.k):
            self.centroids[i] = np.mean(X[labels == i], axis=0)
    
    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute sum of squared distances of each point to the nearest centroid.

        Args:
        - X (np.ndarray): Input array containing data.
        - labels (np.ndarray): Array containing cluster labels assigned to each point.

        Returns:
        - float: Sum of squared distances of each point to the nearest centroid.
        """
        distances = np.linalg.norm(X - self.centroids[labels], axis=1)
        return np.sum(distances ** 2)