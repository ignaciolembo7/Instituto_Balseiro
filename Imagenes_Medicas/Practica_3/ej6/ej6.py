import numpy as np
import cv2
import sys 
import os.path
from skimage.transform import radon, rescale
import matplotlib.pyplot as plt

def read_pgm_file(file_name):

    data_dir = os.path.dirname(os.path.abspath(__file__))

    # Test if file exists
    file_path = os.path.join(data_dir, file_name)
    assert os.path.isfile(file_path), 'file \'{0}\' does not exist'.format(file_path)

    img = cv2.imread(file_name,flags=cv2.IMREAD_GRAYSCALE)

    if img is not None:
        print('img.size: ', img.shape)
    else:
        print('imread({0}) -> None'.format(file_path))

    return img

if __name__ == "__main__":

    if(len(sys.argv)<2):
        print("Usage: python radon_circulo.py [infile.pgm] [outfile.pgm]")
        exit(1)

    infile = sys.argv[1]
    #outfile = sys.argv[2]
    
    img = read_pgm_file(infile)
    image = np.array(img)

    proyections = [8,16,32,64]

    for p in proyections:

        print(f"Proyection = {p}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

        #ax1.set_title("Original")
        ax1.imshow(image, cmap=plt.cm.Greys_r)

        theta = np.linspace(0.,  180., p, endpoint=False)
        sinogram = radon(image, theta=theta)
        dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]
        #ax2.set_title("Radon transform\n(Sinogram)")
        ax2.set_xlabel("Ángulo de proyección (grados)")
        ax2.set_ylabel("Posición de proyección (pixels)")
        ax2.imshow(sinogram, cmap=plt.cm.Greys_r, extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy), aspect='auto')
        fig.tight_layout()
        fig.savefig(f"sinograma.pdf", pad_inches = 0, bbox_inches='tight')
        fig.savefig(f"sinograma.png", dpi=600, pad_inches = 0, bbox_inches='tight')
        print("Size of sinogram: {}".format(sinogram.shape))

        from skimage.transform.radon_transform import _get_fourier_filter

        filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', 'None']

        for ix, f in enumerate(filters):
            response = _get_fourier_filter(2000, f)
            plt.plot(response, label=f)

        plt.xlim([0, 1000])
        plt.xlabel('frequency')
        plt.legend()

        from skimage.transform import iradon
        reconstruction_fbp = iradon(sinogram, theta=theta, filter_name='ramp')
        error = reconstruction_fbp - image
        print(f'FBP rms reconstruction error: {np.sqrt(np.mean(error**2)):.3g}')

        imkwargs = dict(vmin=-0.2, vmax=0.2)

        #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5), sharex=True, sharey=True)
        fig, ax1 = plt.subplots()
        
        #ax1.set_title("Reconstruction\nFiltered back projection")
        ax1.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
        ax1.axis('off')
        print("Size of reconstruction: {}".format(reconstruction_fbp.shape))

        #ax2.set_title("Reconstruction error\nFiltered back projection")
        #ax2.imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, **imkwargs)

        fig.tight_layout()
        fig.savefig(f"retroproyeccion_p={p}_filter=ramp.pdf", pad_inches = 0,bbox_inches='tight')
        fig.savefig(f"retroproyeccion_p={p}_filter=ramp.png", dpi=600, pad_inches = 0, bbox_inches='tight')