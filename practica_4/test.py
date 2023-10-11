import itertools
import numpy as np

def generate_combinations(length):
    # Genera todas las combinaciones posibles de 1 y -1 con la longitud deseada
    combinations = list(itertools.product([1, -1], repeat=length))
    x = np.array(combinations)
    y = np.prod(x, axis=1)
    return x, y

# Define la longitud del vector
length = 5

# Genera todas las combinaciones de 1 y -1
x, y = generate_combinations(length)

# Convierte las combinaciones en un vector de arrays de NumPy


x1 = x[0].reshape(-1, 1)
y1 = y[0].reshape(-1, 1)
print(x1,y1)
