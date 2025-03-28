from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Callable
from tqdm import tqdm

class BaseEstimator(ABC):
    """
    Base estimator class.

    As a starting point, all estimators are able to make a prediction,
    both supervised and unsupervised ones.
    """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...


class SGDEstimator(BaseEstimator, ABC):
    """
    SGD Estimator base class.
    A specialization of Estimator that is also an abstract class, that is able to get
    a set of coefficients or fit a matrix through a SGD process.
    """
    def __init__(
        self,
        learning_rate: float,
        batch_size: int,
        num_epochs: int,
    ) -> None:
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.coef_ = None  
        self.intercept_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        self.coef_ = np.random.uniform(-0.01,0.01, size = n_features) 
        self.intercept_ = np.random.uniform(-0.01,0.01) 


        # SGD
        for epoch in tqdm(range(self.num_epochs), desc="Training"):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch_indices = indices[start:start + self.batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                gradient_intercept, gradient = self._gradient(X_batch, y_batch)
                self.coef_ -= self.learning_rate * gradient
                self.intercept_ -= self.learning_rate * gradient_intercept

    @abstractmethod
    def _gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray: ...


class GradientBoostingEstimator(BaseEstimator, ABC):
    """
    Gradient Boosting Estimator class.

    This class provides the framework for Gradient Boosting techniques which are
    used for both regression and classification tasks.
    """
    def __init__(
        self,
        n_estimators: int,
        learning_rate: float,
        model_class: Callable[..., BaseEstimator],
        model_kwargs: Dict[str, str],
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.models : List[BaseEstimator] = None 

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        residual = y 
        self.models = [] 
        for _ in tqdm(range(self.n_estimators)):
            model = self._train_new_model(X,residual)
            self.models.append(model)
            residual = self._compute_residuals(X, residual)

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.zeros(X.shape[0])
        for model in self.models:
            predictions += model.predict(X)
        return predictions

    def _compute_residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return y - self.learning_rate*self.models[-1].predict(X)

    def _train_new_model(self, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
        model = self.model_class(**self.model_kwargs)
        model.fit(X, y)
        return model