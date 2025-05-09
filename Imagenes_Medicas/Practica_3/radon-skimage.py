#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 13:58:04 2022

@author: matog
"""
import numpy as np
import matplotlib.pyplot as plt

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale

image = shepp_logan_phantom()
image = rescale(image, scale=0.4, mode='reflect')#, channel_axis=None)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

ax1.set_title("Original")
ax1.imshow(image, cmap=plt.cm.Greys_r)

theta = np.linspace(0., 180., 120, endpoint=False)
sinogram = radon(image, theta=theta)
dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]
ax2.set_title("Radon transform\n(Sinogram)")
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")
ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
           extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
           aspect='auto')

fig.tight_layout()
plt.show()


from skimage.transform.radon_transform import _get_fourier_filter

filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann']

for ix, f in enumerate(filters):
    response = _get_fourier_filter(2000, f)
    plt.plot(response, label=f)

plt.xlim([0, 1000])
plt.xlabel('Frecuencia')
plt.legend()
plt.show()


from skimage.transform import iradon
reconstruction_fbp = iradon(sinogram, theta=theta, filter_name='ramp')
error = reconstruction_fbp - image
print(f'FBP rms reconstruction error: {np.sqrt(np.mean(error**2)):.3g}')

imkwargs = dict(vmin=-0.2, vmax=0.2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
                               sharex=True, sharey=True)
ax1.set_title("Reconstruction\nFiltered back projection")
ax1.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
ax2.set_title("Reconstruction error\nFiltered back projection")
ax2.imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, **imkwargs)
plt.show()

#from skimage.transform import iradon_sart
#
#reconstruction_sart = iradon_sart(sinogram, theta=theta)
#error = reconstruction_sart - image
#print(f'SART (1 iteration) rms reconstruction error: '
#      f'{np.sqrt(np.mean(error**2)):.3g}')
#
#fig, axes = plt.subplots(2, 2, figsize=(8, 8.5), sharex=True, sharey=True)
#ax = axes.ravel()
#
#ax[0].set_title("Reconstruction\nSART")
#ax[0].imshow(reconstruction_sart, cmap=plt.cm.Greys_r)
#
#ax[1].set_title("Reconstruction error\nSART")
#ax[1].imshow(reconstruction_sart - image, cmap=plt.cm.Greys_r, **imkwargs)
#
## Run a second iteration of SART by supplying the reconstruction
## from the first iteration as an initial estimate
#reconstruction_sart2 = iradon_sart(sinogram, theta=theta,
#                                   image=reconstruction_sart)
#error = reconstruction_sart2 - image
#print(f'SART (2 iterations) rms reconstruction error: '
#      f'{np.sqrt(np.mean(error**2)):.3g}')
#
#ax[2].set_title("Reconstruction\nSART, 2 iterations")
#ax[2].imshow(reconstruction_sart2, cmap=plt.cm.Greys_r)
#
#ax[3].set_title("Reconstruction error\nSART, 2 iterations")
#ax[3].imshow(reconstruction_sart2 - image, cmap=plt.cm.Greys_r, **imkwargs)
#plt.show()
#
