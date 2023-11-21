import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

#Ploteo 
import seaborn as sns
#sns.axes_style("whitegrid")
sns.set_style("ticks")

def genera_patrones(p, N):
    return np.array([[random.choice([-1, 1]) for _ in range(N)] for _ in range(p)])

Ns = [500,1000,2000,4000]
alphas = [0.12,0.14, 0.16, 0.18]

fig, axs = plt.subplots(nrows=len(alphas), ncols=len(Ns), figsize=(10, 6))

j = 0
for N in Ns:
    idx = np.arange(N)
    l = 0
    for alpha in alphas:

        #Generacion de patrones
        x = genera_patrones(int(N*alpha), N)
        #Matriz de conexiones
        w = np.zeros((N,N))
        #Vector de overlaps
        m = np.zeros(int(alpha*N))

        for u in range(int(alpha*N)):
            w += (1/N)*np.dot(x[u].reshape(-1,1), x[u].reshape(1,-1))
        np.fill_diagonal(w, 0)

        #Secuencial
        print("Secuencial - N=", N, " - alpha=", alpha) 
        for u in tqdm(range(int(alpha*N))):
            s = x[u].copy()
            f = True
            while f:
                r = 0 
                #np.random.shuffle(idx)
                f = False
                for i in idx:
                    h = np.sign(np.dot(w[i], s))
                    if(s[i]*h < 0):
                        f = True
                    s[i] = h
                #if(r==1000):
                #    f = False
                #    print("Corto por límite de iteraciones")
                #r += 1

            #overlap
            m[u] = (1/N)*np.dot(x[u],s)

        bin_width = 0.1  # Ancho constante de los bins
        bin_edges = np.arange(0, 1 + bin_width, bin_width)
        axs[l,j].hist(m, bins=bin_edges, alpha=1, color='b')
        axs[l,j].set_title(f"N = {N}, $\\alpha$ = {alpha}", fontsize = 9)
        axs[l,j].tick_params(direction='in', top=True, right=True, left=True, bottom=True)
        axs[l,j].tick_params(axis='x', rotation=0, labelsize=10, color='black')
        axs[l,j].tick_params(axis='y', labelsize=10, color='black')
        axs[l,j].set_xlim(0, 1.1)
        if(l != len(Ns)-1 ):
            axs[l,j].axes.xaxis.set_ticklabels([])
        #if(j != 0):
            #axs[l,j].axes.yaxis.set_ticklabels([])
    
        """
        #Paralelo
        m = np.zeros(int(alpha*N))
        print("Paralelo - N=", N, " - alpha=", alpha)
        for u in tqdm(range(int(alpha*N))):
            s_j = x[u].copy()
            s_i = s_j.copy()
            f = True 
            idx = 0
            while f:
                s_i = np.sign(np.dot(w, s_j.reshape(-1,1)))
                if((s_i == s_j).all() or idx == 1000):
                    f = False
                s_j = s_i
                idx += 1    
            print("Alcanzó el límite de iteraciones")
        """
        l = l + 1
    j = j + 1 

for j in range(len(Ns)):
    axs[len(Ns)-1,j].set_xlabel("Overlap $m$")
for l in range(len(alphas)):
    axs[l,0].set_ylabel("P(m)")

fig.savefig(f"../Redes-Neuronales/Practica_6/resultados/ej1/hist_sec.pdf")
fig.savefig(f"../Redes-Neuronales/Practica_6/resultados/ej1/hist_sec.png", dpi=600)
plt.show()