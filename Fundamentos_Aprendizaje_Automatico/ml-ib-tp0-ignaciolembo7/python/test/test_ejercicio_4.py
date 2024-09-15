import numpy as np
from ..src.ejercicio_4 import stats, matmul, eigen
import pytest

def test_stats():
    media, desviacion_std = stats(np.array([1, 2, 3, 4, 5]))
    np.testing.assert_almost_equal(media, 3.0)
    np.testing.assert_almost_equal(desviacion_std, 1.41421356237)

def test_matmul():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[2, 0], [1, 2]])
    result = matmul(a, b)
    expected = np.array([[4, 4], [10, 8]])
    np.testing.assert_array_equal(result, expected)

    # Test para ValueError
    c = np.array([1, 2,3])
    with pytest.raises(ValueError):
        matmul(a, c)

def test_eigen():
    a = np.array([[1, 2], [2, 3]])
    valores, vectores = eigen(a)
    assert valores is not None
    assert vectores is not None

    # Test para ValueError
    b = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        eigen(b)
