import pandas as pd
import numpy as np

from ej3.ej3 import histogram, one_hot_encoder

def test_histogram():
    series = pd.Series([1, 2, 5, 3, 4, 2, 6])
    assert all(np.histogram(series, bins=10) == histogram(series))

def test_one_hot_encoder():
    series = pd.Series([1, 2, 5, 3, 4, 2, 6])
    assert all(pd.get_dummies(series) == one_hot_encoder(series))
