import numpy as np
import matplotlib.pyplot as plt
import HH_neuron as hh
import rk4 as rk4
import seaborn as sns
from scipy.signal import find_peaks

#sns.set(context='paper')
sns.axes_style("whitegrid")
sns.set_style("ticks")
fig1, ax1 = plt.subplots(figsize=(8,6)) 
fig2, ax2 = plt.subplots(figsize=(8,6)) 

for Vsyn in [0, -80]:
    gsyns = np.linspace(0, 1, 20)
    f = []
    shift = []
    hh.Vsyn = Vsyn
    print(hh.Vsyn)
    for gsyn in gsyns:
        hh.gsyn = gsyn
        print(hh.gsyn)
        # Valores iniciales: y = [s1, s2, n1, n2, m1, m2, h1, h2, V1, V2]
        y_initial = np.array([hh.s0(hh.Vk), hh.s0(hh.Vk), hh.n0(hh.Vk), hh.n0(hh.Vk), hh.m0(hh.Vk), hh.m0(hh.Vk), hh.h0(hh.Vk), hh.h0(hh.Vk), hh.Vk, 0])
        t_initial = 0.0
        t_final = 1500
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
                
        peaks1, _ = find_peaks(V1_values, height=0) 
        peaks2, _ = find_peaks(V2_values, height=0) 
        
        # Encontrar los picos dentro del rango de interés
        peaks1 = peaks1[-10:]
        peaks2 = peaks2[-10:]
        T1 = (t_values[peaks1[1:]] - t_values[peaks1[:-1]])
        T2 = (t_values[peaks2[1:]] - t_values[peaks2[:-1]])

        T = (np.concatenate((T1,T2))).mean()
        f.append(1.0/(T*1e-3)) #Hz
        Tdiff = np.abs(t_values[peaks1] - t_values[peaks2])
        Tshift = Tdiff.mean()
        shift.append((Tshift%T)/T * 2 * np.pi) 

    if(Vsyn == 0):
        color = 'darkorange'
    else:
        color = 'royalblue'

    # Ploteo para la figura 1
    ax1.plot(gsyns, f, "-o", color= color,  label = "$V_{syn}$ = " + str(Vsyn) )
    ax1.set_xlabel("$g_{syn}$ (mS/cm$^2$)", fontsize=15,)
    ax1.set_ylabel("Frecuencia (Hz)", fontsize=15,)
    ax1.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax1.legend(fontsize=15, framealpha=1)
    #ax1.set_xlim(0, 200)
    ax1.grid(True, linewidth=0.5, linestyle='-', alpha=0.9)
    #ax1.annotate('$g_{syn}$ = ' + str(hh.gsyn) + ' mS/cm$^2$ \n$V_{syn} = $' + str(hh.Vsyn) + ' mV', xy=(0.1, 0.825), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle='round,pad=0.2', edgecolor='grey', facecolor='white'))

    # Guardar figura 1
    fig1.savefig(f"../Redes-Neuronales/Practica_2/resultados/frecuencia_vs_gsyn.pdf")
    fig1.savefig(f"../Redes-Neuronales/Practica_2/resultados/frecuencia_vs_gsyn.png", dpi=600)

    # Ploteo para la figura 2
    ax2.plot(gsyns, shift, "-o", color= color, label = "$V_{syn}$ = " + str(Vsyn))
    ax2.set_xlabel("$g_{syn}$ (mS/cm$^2$)", fontsize=15,)
    ax2.set_ylabel("Desfasaje (rad)", fontsize=15,)
    ax2.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax2.set_yticks([0, np.pi, 2*np.pi])
    ax2.set_yticklabels(["$0$", r"$\pi$", r"$2\pi$"])
    ax2.legend(fontsize=15, framealpha=1)
    #ax2.set_xlim(0, 200)
    ax2.grid(True, linewidth=0.5, linestyle='-', alpha=0.9)
    #ax2.annotate('$g_{syn}$ = ' + str(hh.gsyn) + ' mS/cm$^2$ \n$V_{syn} = $' + str(hh.Vsyn) + ' mV', xy=(0.1, 0.825), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle='round,pad=0.2', edgecolor='grey', facecolor='white'))

    # Guardar figura 2
    fig2.savefig(f"../Redes-Neuronales/Practica_2/resultados/desfasaje_vs_gsyn.pdf")
    fig2.savefig(f"../Redes-Neuronales/Practica_2/resultados/desfasaje_vs_gsyn.png", dpi=600)
   
plt.show() 

