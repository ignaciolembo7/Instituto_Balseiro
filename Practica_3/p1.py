import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.optimize import curve_fit

#Ploteo 
import seaborn as sns
#sns.axes_style("whitegrid")
sns.set_style("ticks")

#Carga de datos
stimulus = np.loadtxt(r"../Redes-Neuronales/Practica_3/stimulus.dat")
tiempo = stimulus[:, 0]
estimulo = stimulus[:, 1]
spikes = np.loadtxt(r"../Redes-Neuronales/Practica_3/spikes.dat")
num_experimentos, num_pasos = spikes.shape

# Ploteo para la figura 1
fig1, ax1 = plt.subplots(figsize=(12,6))
ax1.plot(tiempo, estimulo, "-")
ax1.set_xlabel("Tiempo $t$ (ms)", fontsize=18)
ax1.set_ylabel("Estimulo (dB)", fontsize=18)
ax1.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
ax1.tick_params(axis='x',rotation=0, labelsize=18, color='black')
ax1.tick_params(axis='y', labelsize=18, color='black')
ax1.set_xlim(0, 1000)
ax1.grid(True, linewidth=0.5, linestyle='-', alpha=0.9)
# Guardar figura 1
fig1.savefig(f"../Redes-Neuronales/Practica_3/resultados/estimulo_vs_t.pdf")
fig1.savefig(f"../Redes-Neuronales/Practica_3/resultados/estimulo_vs_t.png", dpi=600)

x, y = np.meshgrid(np.arange(num_pasos)*0.1, np.arange(num_experimentos))
# Crear una máscara para valores iguales a 1
mascara = spikes == 1

# Ploteo para la figura 2
fig2, ax2 = plt.subplots(figsize=(12,6)) 
ax2.scatter(x[mascara], y[mascara], marker='|', s=50)
ax2.set_xlabel("Tiempo $t$ (ms)", fontsize=18)
ax2.set_ylabel("Realización", fontsize=18)
ax2.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
ax2.tick_params(axis='x',rotation=0, labelsize=18, color='black')
ax2.tick_params(axis='y', labelsize=18, color='black')
ax2.set_xlim(0, 1000)
ax2.set_ylim(0, num_experimentos)
ax2.grid(True, linewidth=0.5, linestyle='-', alpha=0.9)
# Guardar figura 2
fig2.savefig(f"../Redes-Neuronales/Practica_3/resultados/reailzaciones_vs_t.pdf")
fig2.savefig(f"../Redes-Neuronales/Practica_3/resultados/realizaciones_vs_t.png", dpi=600)

#Ejercicio 1

diffs = []

for i in range(num_experimentos):
    diff = np.diff(np.where(spikes[i, :] == 1)[0]) * 0.1
    diffs.extend(diff)

# Ploteo para la figura 3
fig3, ax3 = plt.subplots(figsize=(8,6)) 
ax3.hist(diffs, bins=75, density=True, edgecolor='black')
ax3.set_xlabel("Tiempo entre disparos ISI (ms)", fontsize=18)
ax3.set_ylabel(r"P(ISI)", fontsize=18)
ax3.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
ax3.tick_params(axis='x',rotation=0, labelsize=18, color='black')
ax3.tick_params(axis='y', labelsize=18, color='black')
ax3.grid(True, linewidth=0.5, linestyle='-', alpha=0.9, zorder=0)
ax3.set_zorder(1)
# Guardar figura 3
fig3.savefig(f"../Redes-Neuronales/Practica_3/resultados/histograma_ISI.pdf")
fig3.savefig(f"../Redes-Neuronales/Practica_3/resultados/histograma_ISI.png", dpi=600)

mean = np.mean(diffs)
print(f'Promedio: {mean:.2f} ms')
std_deviation = np.std(diffs)
print(f'Desviación estándar: {std_deviation:.2f} ms')
CV = std_deviation/mean 
print(f'CV: {CV:.2f}')

#Ejercicio 2

spikes_exp = np.zeros(num_experimentos)
for i in range(num_experimentos):
    spikes_exp[i] = np.sum(spikes[i, :] == 1)

fig4, ax4 = plt.subplots(figsize=(8,6)) 
ax4.hist(spikes_exp, bins=12, density=True, edgecolor='black', alpha=0.7)
ax4.set_xlabel("Número de spikes por realización $N$", fontsize=18)
ax4.set_ylabel(r"P($N$)", fontsize=18)
ax4.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
ax4.tick_params(axis='x',rotation=0, labelsize=18, color='black')
ax4.tick_params(axis='y', labelsize=18, color='black')
ax4.grid(True, linewidth=0.5, linestyle='-', alpha=0.9, zorder = 0)
ax4.set_zorder(1)
# Guardar figura 4
fig4.savefig(f"../Redes-Neuronales/Practica_3/resultados/histograma_spikes.pdf")
fig4.savefig(f"../Redes-Neuronales/Practica_3/resultados/histograma_spikes.png", dpi=600)

# Calcular el promedio y la desviación estándar de la cantidad de spikes por experimento
promedio_spikes = np.mean(spikes_exp)
desviacion_estandar_spikes = np.std(spikes_exp)
print(f'Promedio de spikes por experimento: {promedio_spikes:.2f}')
print(f'Desviación Estándar de spikes por experimento: {desviacion_estandar_spikes:.2f}')
F = (desviacion_estandar_spikes**2)/promedio_spikes
print(f'Factor de Fano: {F:.2f}')

#Ejercicio 3
w = 200
tasa_de_disparo = np.zeros(len(tiempo))

for t in range(len(tiempo)-w):
    inicio = t  
    fin = inicio + w
    tasas_de_disparo_en_ventana = np.zeros(num_experimentos)

    for i in range(num_experimentos):
        tasas_de_disparo_en_ventana[i] = (np.sum(spikes[i, inicio:fin] == 1)/w)*10000
    
    tasa_de_disparo[t] = np.mean(tasas_de_disparo_en_ventana)
         
# Ploteo para la figura 5
fig5, ax5 = plt.subplots(figsize=(8,6)) 
ax5.plot(tiempo, tasa_de_disparo, "-")
ax5.set_xlabel(r"Tiempo $t$ (ms)", fontsize=18)
ax5.set_ylabel(r"Tasa de disparo $r(t)$ (Hz)", fontsize=18)
ax5.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
ax5.tick_params(axis='x',rotation=0, labelsize=18, color='black')
ax5.tick_params(axis='y', labelsize=18, color='black')
ax5.set_xlim(0, 980)
ax5.grid(True, linewidth=0.5, linestyle='-', alpha=0.9)
# Guardar figura 5
fig5.savefig(f"../Redes-Neuronales/Practica_3/resultados/tasa_disparo.pdf")
fig5.savefig(f"../Redes-Neuronales/Practica_3/resultados/tasa_disparo.png", dpi=600)

# Ploteo para la figura 6
fig6, ax6 = plt.subplots(figsize=(8,6)) 
ax6.hist(tasa_de_disparo, bins=30, density=True, edgecolor='black', alpha=0.7)
ax6.set_xlabel("Tasa de disparo", fontsize=18)
ax6.set_ylabel(r"P($r$)", fontsize=18)
ax6.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
ax6.tick_params(axis='x',rotation=0, labelsize=18, color='black')
ax6.tick_params(axis='y', labelsize=18, color='black')
ax6.grid(True, linewidth=0.5, linestyle='-', alpha=0.9, zorder = 0)
ax6.set_zorder(1)
# Guardar figura 6
fig6.savefig(f"../Redes-Neuronales/Practica_3/resultados/histograma_tasa_disparo.pdf")
fig6.savefig(f"../Redes-Neuronales/Practica_3/resultados/histograma_tasa_disparo.png", dpi=600)

tau_range = np.arange(0, 10002, 1) 
C = np.zeros(len(tau_range))   

for i in range(num_experimentos):
    #print(i)
    spikes_indices = np.where(spikes[i, :] == 1)[0]
    for tau in tau_range:
        C_exp = 0
        for spike_index in spikes_indices:
            t_i = spike_index
            if t_i > tau:
                s = stimulus[t_i - tau][1]
                C_exp += s
        C[tau] += C_exp/(len(spikes_indices))

var = np.var(estimulo)

C = C/num_experimentos
D = C/var

# Ploteo para la figura 7
fig7, ax7 = plt.subplots(figsize=(12,6)) 
ax7.plot(tau_range*0.1, D, "-")
ax7.set_xlabel(r"Tiempo $\tau$ (ms)", fontsize=18)
ax7.set_ylabel(r"Filtro lineal $D(\tau)$ (Hz dB$^-1$ ms$^-1$)", fontsize=18)
ax7.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
ax7.tick_params(axis='x',rotation=0, labelsize=18, color='black')
ax7.tick_params(axis='y', labelsize=18, color='black')
ax7.set_xlim(-25, 1000)
ax7.grid(True, linewidth=0.5, linestyle='-', alpha=0.9)
# Ploteo para la figura 7
fig7.savefig(f"../Redes-Neuronales/Practica_3/resultados/filtro_lineal.pdf")
fig7.savefig(f"../Redes-Neuronales/Practica_3/resultados/filtro_lineal.png", dpi=600)