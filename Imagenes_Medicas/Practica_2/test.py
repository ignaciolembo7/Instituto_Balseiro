import numpy as np

def media_cuadrados(matriz):
    matriz_cuadrada = np.square(matriz)
    suma_total = np.sum(matriz_cuadrada)
    media = suma_total / matriz.size
    return media

# Ejemplo de uso
im = np.array([[1, 2], [3,4]])

resultado = media_cuadrados(im)
print("La media de los cuadrados de las entradas de la matriz es:", resultado)