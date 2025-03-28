import numpy as np
import pandas as pd
import pytest
from ej1.k_means import KMeans

def test_kmeans_centroids():
    
    k = 4
    seed = 420
    n_iter = 100
    tolerance = 0.05  # Margen de error permitido

    expected_centroids = np.array([
        [ 0.70969128, 13.05983087],
        [ 5.42267431, 14.25064546],
        [ 2.33897332,  2.2250205 ],
        [10.11858844,  9.89610836]])

    data = pd.read_csv('ej1/data/synthetic_dataset_1.csv')
    X = data[['x', 'y']].values
    model = KMeans(k=k, seed = seed)
    model.fit(X, n_iter) 

    assert model.centroids == pytest.approx(expected_centroids, abs=tolerance)