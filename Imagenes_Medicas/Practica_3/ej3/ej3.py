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

def save_img_hist(im,filename):
    
    vmin = np.amin(im)
    vmax = np.max(im)
    print("Intensity Min: {}  Max:{}".format(vmin,vmax))

    L = vmax - vmin
    print("Number of Levels: {}".format(L))
    fig = plt.figure(figsize=(12,5))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    # imgplot = plt.imshow(im/np.amax(im))
    imgplot = ax1.imshow(im,cmap='gray', vmin=vmin, vmax=vmax)
    fig.colorbar(imgplot, ax=ax1)
    #ax1.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax1.tick_params(axis='x', rotation=0, labelsize=15, color='black')
    ax1.tick_params(axis='y', labelsize=15, color='black')
    # cv2.imshow(infile,img)
    # cv2.waitKey(0)

    hist, bin_edges = np.histogram(im.ravel(),bins=L)
    ax2.bar(bin_edges[:-1], hist, alpha=1, color='b')
    ax2.set_xlabel(r"Intensidad de pixel", fontsize=15)
    ax2.set_ylabel(r"Número de píxeles", fontsize=15)
    ax2.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax2.tick_params(axis='x', rotation=0, labelsize=15, color='black')
    ax2.tick_params(axis='y', labelsize=15, color='black')
    ax2.set_xlim(-5, 260)
    #ax2.set_ylim(0, 1000)

    #x1 = rmin  # Punto inicial en el eje x
    #x2 = rmax  # Punto final en el eje x
    #y1 = 0  # Punto inicial en el eje y (puedes ajustar este valor según tu necesidad)
    #y2 = 255  # Punto final en el eje y (puedes ajustar este valor según tu necesidad)

    #Agrega la recta al segundo subplot (ax2)
    #ax2.plot([x1, x2], [y1, y2], color='r', linestyle='-', linewidth=2)  # Trama la línea recta

    fig.savefig(f"{filename}_hist.pdf")
    fig.savefig(f"{filename}_hist.png", dpi=600)
    #plt.show()

def save_img(im,filename):
    
    vmin = np.amin(im)
    vmax = np.max(im)
    print("Intensity Min: {}   Max:{}".format(vmin,vmax))

    L = vmax - vmin
    print("Number of Levels: {}".format(L))

    fig, ax = plt.subplots()
    imgplot = ax.imshow(im,cmap='gray', vmin=vmin, vmax=vmax)
    ax.axis('off')

    fig.savefig(f"{filename}.pdf", pad_inches = 0,bbox_inches='tight')
    fig.savefig(f"{filename}.png", dpi=600, pad_inches = 0, bbox_inches='tight')
    print("Size of image: {}".format(im.shape))
    #plt.show()

def sustraction(im1,im2):
    imout = im1 - im2
    return imout

def reconstruct_error(im1, im2):
    dif = im1/np.mean(im1) - im2/np.mean(im2) #normalizo la media
    error = np.sqrt(np.mean(dif**2))
    print(f"Error de reconstruccion es: {error:.6g}")

if __name__ == "__main__":
    
    if(len(sys.argv)<3):
        print("Usage: python sustraction.py [infile.pgm] [outfile.pgm]")
        exit(1)

    infile1 = sys.argv[1]
    infile2 = sys.argv[2]
    outfile1 = sys.argv[3]
    
    img1 = read_pgm_file(infile1)
    img2 = read_pgm_file(infile2)
    im1 = np.array(img1)
    im2 = np.array(img2)
    
    print("\n Sustraction")
    imout = sustraction(im1,im2)
    reconstruct_error(im1,im2)
    save_img(imout,f"sustraction_{infile1}_{infile2}")
    cv2.imwrite(outfile1,imout,[cv2.IMWRITE_PXM_BINARY,0])