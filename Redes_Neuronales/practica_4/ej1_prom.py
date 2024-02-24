import numpy as np
import matplotlib.pyplot as plt

#Ploteo 
import seaborn as sns
#sns.axes_style("whitegrid")
sns.set_style("ticks")

fig1, ax1 = plt.subplots(figsize=(8,6)) 
fig2, ax2 = plt.subplots(figsize=(8,6)) 

dataa = np.loadtxt(r"../Redes-Neuronales/Practica_4/resultados/ej1a/prom_ej1a.txt")
datab = np.loadtxt(r"../Redes-Neuronales/Practica_4/resultados/ej1b/prom_ej1b.txt")

epochsa = dataa[:, 0]
loss_proma = dataa[:, 1]
accs_proma = dataa[:, 2]

epochsb = datab[:, 0]
loss_promb = datab[:, 1]
accs_promb = datab[:, 2]


ax1.semilogx(epochsa, loss_proma, "-", label = "Arquitectura 2-2-1")
ax1.semilogx(epochsb, loss_promb, "-", label = "Arquitectura 2-1-1")
ax1.set_xlabel(r"Época", fontsize=18)
ax1.set_ylabel(r"Error cuádratico medio", fontsize=18)
ax1.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
ax1.tick_params(axis='x', rotation=0, labelsize=18, color='black')
ax1.tick_params(axis='y', labelsize=18, color='black')
#ax1.set_xlim(0, 10000)
ax1.text(0.05, 0.95, r'A' , transform=ax1.transAxes, fontsize=24, verticalalignment='top', fontweight='bold', color="black")
ax1.legend(fontsize=15, framealpha=1)
ax1.grid(True, linewidth=0.5, linestyle='-', alpha=0.9)
fig1.savefig(f"../Redes-Neuronales/Practica_4/resultados/mseprom_ej1.pdf")
fig1.savefig(f"../Redes-Neuronales/Practica_4/resultados/mseprom_ej1.png", dpi=600)

ax2.semilogx(epochsa, accs_proma, "-", label = "Arquitectura 2-2-1" )
ax2.semilogx(epochsb, accs_promb, "-", label = "Arquitectura 2-1-1")
ax2.set_xlabel(r"Época", fontsize=18)
ax2.set_ylabel(r"Precisión media", fontsize=18)
ax2.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
ax2.tick_params(axis='x', rotation=0, labelsize=18, color='black')
ax2.tick_params(axis='y', labelsize=18, color='black')
#ax2.set_xlim(0, 10000)
ax2.legend(fontsize=15, framealpha=1, loc = "center left")
ax2.text(0.05, 0.95, r'B' , transform=ax2.transAxes, fontsize=24, verticalalignment='top', fontweight='bold', color="black")
ax2.grid(True, linewidth=0.5, linestyle='-', alpha=0.9)
fig2.savefig(f"../Redes-Neuronales/Practica_4/resultados/accprom_ej1.pdf")
fig2.savefig(f"../Redes-Neuronales/Practica_4/resultados/accprom_ej1.png", dpi=600)