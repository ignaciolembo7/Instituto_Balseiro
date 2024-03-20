import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, rescale, resize
from skimage.transform.radon_transform import _get_fourier_filter
from tqdm import tqdm

#proyecciones = list(range(20, 800, 20))
proyeccion = 320
#detectores = 367
detector = 367
filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann'] #None
errores = []

for f in tqdm(filters):

    # Escalar y luego redimensionar la imagen
    image = shepp_logan_phantom()
    #image = rescale(image, scale=0.4, mode='reflect')
    shape = (detector, detector)  # Nuevo tamaño en píxeles
    image_resized = resize(image, shape, mode='reflect')

    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
    fig, ax = plt.subplots(figsize=(8, 8))
    #ax1.set_title("Original")
    ax.imshow(image_resized, cmap=plt.cm.Greys_r)
    ax.axis('off') 
    fig.savefig(f"original.pdf", pad_inches = 0, bbox_inches='tight')
    fig.savefig(f"original.png", dpi=600, pad_inches = 0, bbox_inches='tight')

    theta = np.linspace(0., 180., proyeccion, endpoint=False)
    sinogram = radon(image_resized, theta=theta)
    dx, dy = 0.5 * 180.0 / max(image_resized.shape), 0.5 / sinogram.shape[0]
    #ax2.set_title("Radon transform\n(Sinogram)")
    #ax2.set_xlabel("Projection angle (deg)")
    #ax2.set_ylabel("Projection position (pixels)")
    #ax2.imshow(sinogram, cmap=plt.cm.Greys_r, extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),aspect='auto')
    #fig.tight_layout()
    #plt.show()

    #for ix, f in enumerate(filters):
    #    response = _get_fourier_filter(2000, f)
    #    plt.plot(response, label=f)

    #plt.xlim([0, 1000])
    #plt.xlabel('frequency')
    #plt.legend()
    #plt.show()

    reconstruction_fbp = iradon(sinogram, theta=theta, filter_name=f)
    dif = reconstruction_fbp/np.mean(reconstruction_fbp) - image_resized/np.mean(image_resized) #normalizo la media
    errores.append(np.sqrt(np.mean(dif**2)))
    #print(f'FBP rms reconstruction error: {np.sqrt(np.mean(error**2)):.3g}')
    
    imkwargs = dict(vmin=-0.2, vmax=0.2)
    fig, ax = plt.subplots(figsize=(8, 8))
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5), sharex=True, sharey=True)
    #ax.set_title("Reconstruction\nFiltered back projection")
    ax.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
    ax.axis('off') 
    fig.savefig(f"error_vs_filters_{f}.pdf", pad_inches = 0, bbox_inches='tight')
    fig.savefig(f"error_vs_filters_{f}.png", dpi=600, pad_inches = 0, bbox_inches='tight')
    #ax.set_title("Reconstruction error\nFiltered back projection")
    #ax.imshow(reconstruction_fbp - image_resized, cmap=plt.cm.Greys_r, **imkwargs)
    #plt.show()
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
    ax.vlines(cfilters[i], 0, errores[i], linestyles='dashed', colors='gray')
ax.set_ylim(0,0.3)
#ax2.set_ylim(0, 1000)
fig.savefig(f"error_vs_filters.pdf", pad_inches = 0,bbox_inches='tight')
fig.savefig(f"error_vs_filters.png", dpi=600, pad_inches = 0, bbox_inches='tight')