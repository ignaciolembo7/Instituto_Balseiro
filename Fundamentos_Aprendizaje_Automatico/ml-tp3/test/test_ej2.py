import numpy as np
import pytest
from sklearn.decomposition import PCA as sklearnPCA
from ej2.pca import PCA

def test_pca_components():
    X = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

    sklearn_pca = sklearnPCA(n_components=3)
    sklearn_pca.fit(X)

    my_pca = PCA(n_components=3)
    my_pca.fit(X)

    X_transformed_sklearn = sklearn_pca.fit_transform(X)
    X_transformed_my_pca = my_pca.fit_transform(X)

    for i in range(len(X_transformed_sklearn)):
        assert X_transformed_my_pca[i] == pytest.approx(X_transformed_sklearn[i], abs=1e-5)
