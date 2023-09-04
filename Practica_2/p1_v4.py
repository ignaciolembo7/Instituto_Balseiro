import numpy as np
import matplotlib.pyplot as plt
import HH_neuron as hh
import rk4 as rk4
import seaborn as sns
from scipy.signal import find_peaks

#sns.set(context='paper')
sns.axes_style("whitegrid")
sns.set_style("ticks")

gsyns = np.linspace(0, 2, 20)
f = []
shift = []
fig1, ax1 = plt.subplots(figsize=(8,6)) 
fig2, ax2 = plt.subplots(figsize=(8,6)) 
for Vsyn in [0, -80]:

    hh.Vsyn = Vsyn
    print(hh.Vsyn)
    for gsyn in gsyns:
        hh.gsyn = gsyn
        print(hh.gsyn)
        # Valores iniciales: y = [s1, s2, n1, n2, m1, m2, h1, h2, V1, V2]
        y_initial = np.array([hh.s0(hh.Vk), hh.s0(hh.Vk), hh.n0(hh.Vk),  hh.n0(hh.Vk), hh.m0(hh.Vk), hh.m0(hh.Vk), hh.h0(hh.Vk), hh.h0(hh.Vk), hh.Vk, 0])
        t_initial = 0.0
        t_final = 200
        time_step = 0.01

        #Simulacion
        steps = int(t_final/time_step) 
        t_values = np.empty(steps)
        V1_values = np.empty(steps)
        V2_values = np.empty(steps)

        t = t_initial
        y = y_initial

        for i in range(steps):
            # Aplicar un paso de Runge-Kutta de orden 4
            y = rk4.rk4(hh.funcs, y, t, time_step)

            t_values[i] = t
            V1_values[i] = y[8]
            V2_values[i] = y[9]
            
            t += time_step
        
        #Encontrar los picos en la curva
        t_range = t_values[int(steps/2):steps]
        V1_range = V1_values[int(steps/2):steps]
        V2_range = V2_values[int(steps/2):steps]
        # Encontrar los picos dentro del rango de interés
        peaks1, _ = find_peaks(V1_values, height=0, distance=10) 
        peaks2, _ = find_peaks(V2_values, height=0, distance=10) 

        if(len(peaks1) > 2):
            T1 = t_values[peaks1[-1]] - t_values[peaks1[-2]]
        else:
            T1 = 0
        if(len(peaks2) > 2):
            T2 = t_values[peaks1[-1]] - t_values[peaks1[-2]]
        else:
            T2 = 0

        T = np.mean([T1,T2])
        f.append(1.0/(T*1e-3)) #Hz
        Tdiff = t_values[peaks1[-5:]] - t_values[peaks2[-5:]]
        Tshift = np.abs(Tdiff.mean())
        shift.append(Tshift/T * 2 * np.pi) 

    #Ploteo
    ax1.plot(gsyns, f, "-o", color='darkorange',  label = "$V_{syn} = " + str(Vsyn) )
    ax2.plot(gsyns, shift, "-o", color='royalblue', label = "$V_{syn} = " + str(Vsyn))
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

