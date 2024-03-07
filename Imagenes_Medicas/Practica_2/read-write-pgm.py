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

def SemilinearTrans(im,rmin,rmax):
    Imin = 0
    Imax = 255
    a = (Imax - Imin)/(rmax-rmin)
    b = Imin - a*rmin  
    imout = a*im + b

    imout = np.where(imout < Imin, Imin, imout)
    imout = np.where(imout > Imax, Imax, imout)
    imout = imout.astype(int)
    return imout

def Equalize(im):
    imout = im 
    vmin = np.amin(im)
    vmax = np.max(im)
    L = vmax - vmin
    hist, bin_edges = np.histogram(im.ravel(),bins=L)
    hist_norm = hist / np.sum(hist)  #Normaliza el histograma

    suma = 0
    for i in range(0,im.shape[0]):
        for j in range(0,im.shape[1]):
            suma = 0
            for e in range(0,im[i,j]):
               suma += hist_norm[e]
            imout[i,j] = 255*suma

    imout = imout.astype(int)
    return imout

def Trans_binary(im):
    imout=np.where(im<128,255,0)
    imout = imout.astype(int)
    return imout

def Trans_exponential(im,gamma):
    imout=im
    imout=255*(imout/im.max())**gamma
    imout=imout.astype(int)
    return imout

def sustraction(im1,im2):
    imout = im1 - im2
    return imout

def interpolate_nn(im,f):
    Nx, Ny = im.shape[0],im.shape[1]
    Mx=int(f*Nx)
    My=int(f*Ny)
    imout=np.ndarray((Mx,My), dtype=int)

    for x in range(Mx):
        for y in range(My):
            imout[x][y]=im[round(x*Nx/Mx)%Nx][round(y*Ny/My)%Ny]

    return imout

def interpolate_bilinear(im, output_width, output_height):

    # Calculate scaling ratios
    old_height, old_width = im.shape[:2]
    x_ratio = old_width / output_width
    y_ratio = old_height / output_height

    # Create new image array
    new_img = np.zeros((output_height, output_width), dtype=np.uint8)

    # Perform bilinear interpolation
    for y in range(output_height):
        for x in range(output_width):
            x_original = x * x_ratio
            y_original = y * y_ratio
            x0 = int(x_original)
            y0 = int(y_original)
            x1 = min(x0 + 1, old_width - 1)
            y1 = min(y0 + 1, old_height - 1)

            Q11 = im[y0, x0]
            Q21 = im[y1, x0]
            Q12 = im[y0, x1]
            Q22 = im[y1, x1]

            x_weight = x_original - x0
            y_weight = y_original - y0

            R1 = Q11 * (1 - x_weight) + Q12 * x_weight
            R2 = Q21 * (1 - x_weight) + Q22 * x_weight

            P = R1 * (1 - y_weight) + R2 * y_weight

            new_img[y, x] = P

    return new_img

def high_boost(im,im_filtered,A):
    imout = A*im - im_filtered
    return imout

def unsharp(im,im_filtered):
    imout = im - im_filtered
    return imout

if __name__ == "__main__":
    
    if(len(sys.argv)<3):
        print("Usage: python read-write-pgm.py [infile.pgm] [outfile.pgm]")
        exit(1)

    infile1 = sys.argv[1]
    outfile1 = sys.argv[2]
    #infile2 = sys.argv[3]
    #outfile2 = sys.argv[4]
    
    img1 = read_pgm_file(infile1)
    #img2 = read_pgm_file(infile2)
    im1 = np.array(img1)
    #im2 = np.array(img2)

    #Brute image 
    print("\n Brute image:")
    print("Size of image 1: {}".format(im1.shape))
    #print("Size of image 2: {}".format(im2.shape))
    save_img_hist(im1,"ImageA_brute",0,150)

    #Semilinear transformation for different rmin values
    print("\n Semilinear transformation")
    rmax = 150 # 25, 62 , 113 ,150
    rmin = 113 # 0, 25 , 62 , 113 
    imout = SemilinearTrans(im,rmin,rmax)
    print("Size of image: {}".format(imout.shape))
    save_img_hist(imout,"ImageA_"+str(rmin)+"_"+str(rmax))
    save_img(imout,"ImageA_"+str(rmin)+"_"+str(rmax))

    #Equalize image
    print("\n Equalize image")
    imout = Equalize(im1)
    #save_img_hist(imout,"ImageA_EQ")    
    save_img(imout,"ImageA_EQ")    
    cv2.imwrite(outfile1,imout,[cv2.IMWRITE_PXM_BINARY,0])

    #Transformación binaria
    print("\n Transformación binaria")
    imout = Trans_binary(im1)
    save_img(imout,f"ImageA_binary")
    cv2.imwrite(outfile1,imout,[cv2.IMWRITE_PXM_BINARY,0])

    #Transformación exponencial
    print("\n Transformación exponencial")
    gamma = 0.5
    imout = Trans_exponential(im1,gamma)
    save_img(imout,f"ImageA_exp_gamma={gamma}")
    cv2.imwrite(outfile1,imout,[cv2.IMWRITE_PXM_BINARY,0])

    #Interpolate nn
    f = 0.25
    print("\n Interpolate NN")
    imout = interpolate_nn(im1,f)
    save_img(imout,f"Interpolate_nn_f={f}")

    #Interpolate bilinear
    #f = 0.25
    print("\n Interpolate bilinear")
    imout = interpolate_bilinear(im1, int(f* im1.shape[0]), int(f* im1.shape[1]))
    save_img(imout,f"Interpolate_bilinear_f={f}")
    cv2.imwrite(outfile1,imout,[cv2.IMWRITE_PXM_BINARY,0])

    #Sustraction
    print("\n Sustraction")
    imout = sustraction(im1,im2)
    save_img(imout,f"sustraction_exp_gamma=1.75")
    cv2.imwrite(outfile1,imout,[cv2.IMWRITE_PXM_BINARY,0])

    #Unsharp filter
    print("\n Unsharp filter")
    imout1 = unsharp(im1,im2)
    imout1 = np.array(imout1)
    print("Size of image 1: {}".format(imout1.shape))
    save_img(imout1,"unsharp")

    #HighBoost filter
    print("\n HighBoost filter ")
    A=3
    imout2 = high_boost(im1,im2,A)
    imout2 = np.array(imout2)
    print("Size of image 2: {}".format(imout2.shape))
    save_img(imout2,f"highboost_A={A}")

    #High Boost filter plot
    print("\n High Boost filterplot")
    A = np.arange(0.5, 10.5, 0.5)
    S = ([])

    for a in A:
        imout2 = high_boost(im1,im2,a)
        im_sust = sustraction(imout2,im1)
        matriz_cuadrada = np.square(im_sust)
        suma_total = np.sum(matriz_cuadrada)
        media = np.sqrt(suma_total / im_sust.size)
        S = np.append(S, media)
        #imout2 = np.array(imout2)
        #print("Size of image 1: {}".format(imout2.shape))
        #imout2 = Equalize_pgm_file(imout2)
        #save_img(imout2,f"highboost_A={A}")
        #cv2.imwrite(outfile2,imout2,[cv2.IMWRITE_PXM_BINARY,0])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(A, S, linewidth=2)
    ax.set_xlabel(f"Intensidad del filtro HighBoost $A$", fontsize=15)
    ax.set_ylabel(f"Suma cuadratica media", fontsize=15)
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x', rotation=0, labelsize=15, color='black')
    ax.tick_params(axis='y', labelsize=15, color='black')
    #ax.set_xlim(-5, 260)
    #ax2.set_ylim(0, 1000)
    fig.savefig(f"Highboost_plot.pdf", pad_inches = 0,bbox_inches='tight')
    fig.savefig(f"Highboost_plot.png", dpi=600, pad_inches = 0, bbox_inches='tight')
