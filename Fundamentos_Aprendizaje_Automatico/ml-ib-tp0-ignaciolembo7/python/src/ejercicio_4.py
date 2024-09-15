# Ejercicio 4: NumPy
import numpy as np

# Estadísticas de un arreglo
def stats(a):
    a = np.array(a)
    return np.mean(a), np.std(a)

# Producto matricial
def matmul(a, b):
    a = np.array(a)
    b = np.array(b)
    try:
        result = np.matmul(a, b)
        return result
    except ValueError:
        raise ValueError("Las matrices no son compatibles para multiplicación")

# Autovectores y autovalores
def eigen(a):
    a = np.array(a)
    if a.shape[0] != a.shape[1]:
        raise ValueError("La matriz no es cuadrada")
    eigenvalues, eigenvectors = np.linalg.eig(a)
    return eigenvalues, eigenvectors


