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

def signo_T(h, T):
    p = np.exp(h/T) / (np.exp(h/T)+np.exp(-h/T))
    q = 1 - p
    s = np.random.choice([1, -1], p=[p, q])
    return s

N = 4000
p = 40
Ts = np.arange(0.1, 2.1, 0.1)
its = 10

#Vector de overlaps
m_p = np.zeros(len(Ts))
m_std = np.zeros(len(Ts))
fig1, ax1 = plt.subplots(figsize=(8,6))

#Generacion de patrones
x = genera_patrones(p, N)
#Matriz de conexiones
w = np.zeros((N,N))
for u in range(p):
    w += (1/N)*np.dot(x[u].reshape(-1,1), x[u].reshape(1,-1))
np.fill_diagonal(w, 0)

t = 0
idx = np.arange(N)
for T in tqdm(Ts):
    m = np.zeros(p)
    for u in range(p):
        s = x[u].copy()
        for it in range(its):
            ##np.random.shuffle(idx)
            for i in idx:
                h = np.dot(w[i,:], s)
                s[i] = signo_T(h, T)
        #overlap
        m[u] = (1/N)*np.dot(s,x[u])
    m_p[t] = np.mean(m)
    m_std[t] = np.std(m)
    t += 1

ax1.errorbar(Ts, m_p, yerr=m_std, fmt='o')
ax1.fill_between(Ts, m_p+m_std, m_p-m_std, alpha=0.5, color="blue")
ax1.set_xlabel(r"Temperatura T", fontsize=18)
ax1.set_ylabel(r"Overlap promedio $\overline{m}$", fontsize=18)
ax1.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
ax1.tick_params(axis='x', rotation=0, labelsize=18, color='black')
ax1.tick_params(axis='y', labelsize=18, color='black')

fig1.savefig(f"../Redes-Neuronales/Practica_6/resultados/ej2/m_vs_T.pdf")
fig1.savefig(f"../Redes-Neuronales/Practica_6/resultados/ej2/m_vs_T.png", dpi=600)

plt.show()

#ax1.legend(fontsize=12, framealpha=1, loc= "center right")
#ax1.set_xlim(0, num_epochs)
#ax1.text(0.05, 0.9, f'{fig_index}' , transform=ax1.transAxes, fontsize=24, verticalalignment='top', fontweight='bold', color="black")
#ax1.grid(True, linewidth=0.5, linestyle='-', alpha=0.9)