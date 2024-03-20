import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, rescale, resize
from skimage.transform.radon_transform import _get_fourier_filter
from tqdm import tqdm

def set_column_to_zero(sinogram, detector_index):
    sinogram[detector_index, :] = 0

proyeccion = 320
detector = 367
filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann'] #None

# Escalar y luego redimensionar la imagen
image = shepp_logan_phantom()
#image_resized = rescale(image, scale=0.4, mode='reflect')
shape = (detector, detector)  # Nuevo tamaño en píxeles
image_resized = resize(image, shape, mode='reflect')

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(image_resized, cmap=plt.cm.Greys_r)
ax.axis('off') 
fig.savefig(f"original.pdf", pad_inches = 0, bbox_inches='tight')
fig.savefig(f"original.png", dpi=600, pad_inches = 0, bbox_inches='tight')
print("Size of original: {}".format(image_resized.shape))

theta = np.linspace(0., 360., proyeccion, endpoint=False)
sinogram = radon(image_resized, theta=theta)

# Establecer una columna del sinograma en cero (simulación de falta de datos)
detector_roto = 150
set_column_to_zero(sinogram, detector_roto)

fig, ax = plt.subplots(figsize=(8, 8))
dx, dy = 0.5 * 360.0 / max(image_resized.shape), 0.5 / sinogram.shape[0]
sinogram_rot = np.rot90(sinogram)
ax.imshow(sinogram_rot, cmap=plt.cm.Greys_r, extent=(-dx, 360.0 + dx, -dy, sinogram.shape[0] + dy), aspect='auto')
ax.axis('off') 
fig.savefig(f"sinograma_{detector_roto}.pdf", pad_inches = 0, bbox_inches='tight')
fig.savefig(f"sinograma_{detector_roto}.png", dpi = 600, pad_inches = 0, bbox_inches='tight')
print("Size of sinogram: {}".format(sinogram.shape))

reconstruction_fbp = iradon(sinogram, theta=theta, filter_name='ramp')
dif = reconstruction_fbp/np.mean(reconstruction_fbp) - image_resized/np.mean(image_resized) #normalizo la media
error = np.sqrt(np.mean(dif**2))
#print(f'FBP rms reconstruction error: {np.sqrt(np.mean(error**2)):.3g}')
print("Size of reconstruction: {}".format(reconstruction_fbp.shape))
imkwargs = dict(vmin=-0.2, vmax=0.2)
fig, ax = plt.subplots()
ax.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
ax.axis('off') 
fig.savefig(f"reconstruction_{detector_roto}_360.pdf", pad_inches = 0, bbox_inches='tight')
fig.savefig(f"reconstruction_{detector_roto}_360.png", dpi =600, pad_inches = 0, bbox_inches='tight')
plt.close()

