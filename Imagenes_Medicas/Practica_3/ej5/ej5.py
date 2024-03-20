import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, rescale, resize
from skimage.transform.radon_transform import _get_fourier_filter
from tqdm import tqdm

def add_gaussian_noise(image, mean, std_dev):
    noise = np.random.normal(mean, std_dev, size=image.shape)
    noisy_image = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return noisy_image

proyeccion = 320
detector = 367
filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann'] #None
errores = []

for f in tqdm(filters):

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

    theta = np.linspace(0., 180., proyeccion, endpoint=False)
    sinogram = radon(image_resized, theta=theta)

    # Parámetros para el ruido gaussiano en el sinograma
    mean_sinogram = 0  # Media
    sigma=0.1
    std_dev_sinogram = sigma * np.max(sinogram)  # Desviación estándar (10% del valor máximo del sinograma)

    # Agregar ruido gaussiano al sinograma
    sinogram = add_gaussian_noise(sinogram, mean_sinogram, std_dev_sinogram)

    fig, ax = plt.subplots(figsize=(8, 8))
    dx, dy = 0.5 * 180.0 / max(image_resized.shape), 0.5 / sinogram.shape[0]
    sinogram_rot = np.rot90(sinogram)
    ax.imshow(sinogram_rot, cmap=plt.cm.Greys_r, extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),aspect='auto')
    ax.axis('off') 
    fig.savefig(f"sinograma_noise_{sigma}.pdf", pad_inches = 0, bbox_inches='tight')
    fig.savefig(f"sinograma_noise_{sigma}.png", dpi=600, pad_inches = 0, bbox_inches='tight')
    print("Size of sinogram: {}".format(sinogram.shape))

    reconstruction_fbp = iradon(sinogram, theta=theta, filter_name=f)
    dif = reconstruction_fbp/np.mean(reconstruction_fbp) - image_resized/np.mean(image_resized) #normalizo la media
    errores.append(np.sqrt(np.mean(dif**2)))
    #print(f'FBP rms reconstruction error: {np.sqrt(np.mean(error**2)):.3g}')
    print("Size of reconstruction: {}".format(reconstruction_fbp.shape))
    imkwargs = dict(vmin=-0.2, vmax=0.2)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
    ax.axis('off') 
    fig.savefig(f"reconstruction_{f}.pdf", pad_inches = 0, bbox_inches='tight')
    fig.savefig(f"reconstruction_{f}.png", dpi=600, pad_inches = 0, bbox_inches='tight')
    plt.close()

fig, ax = plt.subplots(figsize=(8, 6))

cfilters = list(range(1, len(errores) + 1))
etiquetas_x = ['{}'.format(i) for i in filters]
ax.set_xticks(cfilters)
ax.set_xticklabels(etiquetas_x)
ax.tick_params(axis='x', rotation=60)
ax.plot(cfilters, errores,'o', linewidth=2)
ax.set_xlabel(f"Tipo de filtro", fontsize=15)
ax.set_ylabel(f"Error de reconstrucción", fontsize=15)
ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
ax.tick_params(axis='x', rotation=0, labelsize=15, color='black')
ax.tick_params(axis='y', labelsize=15, color='black')
for i in range(len(errores)):
    ax.vlines(cfilters[i], 0.8, errores[i], linestyles='dashed', colors='gray')
ax.set_ylim(0.8,2.4)
fig.savefig(f"error_vs_filters_noise.pdf", pad_inches = 0,bbox_inches='tight')
fig.savefig(f"error_vs_filters_noise.png", dpi=600, pad_inches = 0, bbox_inches='tight')