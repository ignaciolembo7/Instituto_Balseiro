import numpy as np

class PCA:
    def __init__(self, n_components: int):
        """
        Initialize PCA model.

        Args:
        - n_components (int): The number of principal components to retain.
        """
        self.n_components_ = n_components
        self.components_ = None
        self.mean_ = None
        self.eigenvalues_ = None
        self.eingenvectors_ = None
        self.evr = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fit PCA model to the training data.

        Args:
        - X (np.ndarray): The training data, with shape (n_samples, n_features).
        """
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        self.eigenvalues_, self.eigenvectors_ = np.linalg.eigh(cov_matrix)
        
        sorted_indices = np.argsort(self.eigenvalues_)[::-1]
        self.eigenvalues_ = self.eigenvalues_[sorted_indices]
        self.eigenvectors_ = self.eigenvectors_[:, sorted_indices]
        self.components_ = self.eigenvectors_[:, :self.n_components_]
        
        self.evr_ = np.cumsum(self.eigenvalues_[:self.n_components_]) / np.sum(self.eigenvalues_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform input data using the learned PCA model.

        Args:
        - X (np.ndarray): The input data, with shape (n_samples, n_features).

        Returns:
        - np.ndarray: The transformed data, with shape (n_samples, n_components).
        """
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA model to the training data and transform the input data.

        Args:
        - X (np.ndarray): The input data, with shape (n_samples, n_features).

        Returns:
        - np.ndarray: The transformed data, with shape (n_samples, n_components).
        """
        self.fit(X)
        return self.transform(X)