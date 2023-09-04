import numpy as np
import matplotlib.pyplot as plt
import HH_neuron as hh
import rk4 as rk4
import seaborn as sns
#sns.set(context='paper')
sns.axes_style("whitegrid")
sns.set_style("ticks")

# Valores iniciales: y = [s1, s2, n1, n2, m1, m2, h1, h2, V1, V2]
#y_initial = np.array([s0(Vk), s0(Vk), n0(Vk), n0(Vk), m0(Vk), m0(Vk), h0(Vk)-1, h0(Vk)-1, Vk-1, Vk-1])
y_initial = np.array([hh.s0(hh.Vk), hh.s0(hh.Vk), hh.n0(hh.Vk),  hh.n0(hh.Vk), hh.m0(hh.Vk), hh.m0(hh.Vk), hh.h0(hh.Vk), hh.h0(hh.Vk), hh.Vk, 0])
t_initial = 0.0
t_final = 1000
time_step = 0.01

#Simulacion
steps = int(t_final/time_step) 
t_values = np.empty(steps)
V1_values = np.empty(steps)
V2_values = np.empty(steps)

t = t_initial
y = y_initial
#hh.gsyn = 0.5

for i in range(steps):
    # Aplicar un paso de Runge-Kutta de orden 4
    y = rk4.rk4(hh.funcs, y, t, time_step)

    t_values[i] = t
    V1_values[i] = y[8]
    V2_values[i] = y[9]
    
    t += time_step

#Ploteo
plt.plot(t_values, V1_values, color='darkorange',  label = "Neurona 1")
plt.plot(t_values, V2_values, color='royalblue', label = "Neurona 2")
plt.xlabel("Tiempo (ms)", fontsize=15,)
plt.ylabel("Voltaje (mV)", fontsize=15,)
# Adjust x-axis tick appearance
plt.xticks(rotation=0, fontsize=15, color='black')
# Adjust y-axis tick appearance
plt.yticks(fontsize=15, color='black')
plt.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
plt.legend(fontsize=15, framealpha=1)
plt.xlim(0, 200)  # Establecer límites en el eje x
plt.grid(True, linewidth=0.5, linestyle='-', alpha=0.9)  # Establecer propiedades de la cuadrícula
# Agregar texto en la esquina superior izquierda
#plt.text(0.1, 0.95, '$g_{syn}$ = ' + str(hh.gsyn) + ' mS/cm$^2$ \n$V_{syn} = $' + str(hh.Vsyn) + ' mV', transform=plt.gca().transAxes,fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=1))
plt.annotate('$g_{syn}$ = ' + str(hh.gsyn) + ' mS/cm$^2$ \n$V_{syn} = $' + str(hh.Vsyn) + ' mV', xy=(0.1, 0.825), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle='round,pad=0.2', edgecolor='grey', facecolor='white'))


plt.show() 