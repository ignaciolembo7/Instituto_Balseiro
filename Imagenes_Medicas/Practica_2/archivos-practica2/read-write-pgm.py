import cv2
import sys 
import numpy as np
import matplotlib.pyplot as plt
import os.path



def read_pgm_file(file_name):

    data_dir = os.path.dirname(os.path.abspath(__file__))

    # Test if file exists
    file_path = os.path.join(data_dir, file_name)
    assert os.path.isfile(file_path), 'file \'{0}\' does not exist'.format(file_path)

    img = cv2.imread(file_name,flags=cv2.IMREAD_GRAYSCALE)

    if img is not None:
        print('img.size: ', img.size)
    else:
        print('imread({0}) -> None'.format(file_path))

    return img


def show_img_hist(im):
    
    vmin = np.amin(im)
    vmax = np.max(im)
    print("Intensity Min: {}   Max:{}".format(vmin,vmax))

    L = vmax - vmin
    print("Number of Levels: {}".format(L))
    fig = plt.figure(figsize=(16,6))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    # imgplot = plt.imshow(im/np.amax(im))
    imgplot = ax1.imshow(im,cmap='gray', vmin=vmin, vmax=vmax)
    fig.colorbar(imgplot, ax=ax1)
    # cv2.imshow(infile,img)
    # cv2.waitKey(0)

    hist, bin_edges = np.histogram(im.ravel(),bins=L)
    ax2.bar(bin_edges[:-1], hist)
    plt.show()


def process_pgm_file(im):
    #
    #   Escriba su procesamiento de imágenes aquí
    #
    imout = im
    
    return imout

if __name__ == "__main__":
    
    if(len(sys.argv)<3):
        print("Usage: python read-write-pgm.py [infile.pgm] [outfile.pgm]")
        exit(1)

    infile = sys.argv[1]
    outfile = sys.argv[2]
    
    img = read_pgm_file(infile)

    im = np.array(img)
    print("Size of image: {}".format(im.shape))

    show_img_hist(im)

    imout = process_pgm_file(im)


    cv2.imwrite(outfile,imout,[cv2.IMWRITE_PXM_BINARY,0])
