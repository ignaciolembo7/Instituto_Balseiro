import numpy as np
import cv2
from skimage.transform import radon, rescale
import matplotlib.pyplot as plt

Nx, Ny = 100, 100
image = np.zeros((Nx, Ny), dtype=np.uint8)

# Coordenadas del centro del círculo
centro_x, centro_y = int(Nx/2), int(Ny/2)

# Radio del círculo
radio = 3

# Valor del píxel para el círculo blanco (255 es blanco)
valor_pixel = 255

# Dibujar el círculo en la imagen
cv2.circle(image, (centro_x, centro_y), radio, valor_pixel, -1)

# Guardar la imagen como un archivo PGM
cv2.imwrite('circulo_blanco.pgm', image)
