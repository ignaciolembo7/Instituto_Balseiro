import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#Constantes 
gna = 120 #mS/cm2
gk = 36 #mS/cm2
gcl = 0.3 #mS/cm2

Vna = 50 #mV
Vk = -77 #mV
Vcl = -54.4 #mV

C = 1 #muF/cm2

#Iext = 9 #muA/cm2

a = 0.5
b = 1.5 
gamma = 2
tau = 1
tau_w = 1

def f_w(t, V, w, I):
    # Define la ecuación para dw/dt
    return -gamma*w + b*V

def f_V(t, V, w, I):
    # Define la ecuación para dV/dt
    f = V*(a-V)*(V-1)
    return (f + I - w)/tau_w

# Implementación del método de Runge-Kutta de cuarto orden
def runge_kutta_4_system(t, V, w, I, dt):
    # Calcula los k1, k2, k3 y k4 para cada variable
    k1_w = dt * f_w(t, V, w, I)
    k1_V = dt * f_V(t, V, w, I)
    
    k2_w = dt * f_w(t + dt/2, V + k1_V/2, w + k1_w/2, I)
    k2_V = dt * f_V(t + dt/2, V + k1_V/2, w + k1_w/2, I)
    
    k3_w = dt * f_w(t + dt/2, V + k2_V/2, w + k2_w/2, I)
    k3_V = dt * f_V(t + dt/2, V + k2_V/2, w + k2_w/2, I)
    
    k4_w = dt * f_w(t + dt, V + k3_V, w + k3_w, I)
    k4_V = dt * f_V(t + dt, V + k3_V, w + k3_w, I)
    
    # Calcula las nuevas variables utilizando los k1, k2, k3 y k4
    new_w = w + (k1_w + 2*k2_w + 2*k3_w + k4_w) / 6
    new_V = V + (k1_V + 2*k2_V + 2*k3_V + k4_V) / 6
    
    return new_w, new_V

f = []
I_ext = list(range(1, 5))  # Crear un vector del 1 al 20

for i in I_ext:
    # Parámetros de la simulación
    dt = 0.01  # Tamaño del paso de tiempo
    t_end = 50.0  # Tiempo final de simulación

    # Listas para almacenar resultados
    V0=Vk
    w_values = [0]
    V_values = [V0]
    t_values = [0]

    # Bucle de integración
    t_current = 0
    w_current = V0 
    V_current = V0

    t_vec = []
    while t_current < t_end:
        w_current, V_current = runge_kutta_4_system(t_current, V_current, w_current, i, dt)
        t_current += dt
        
        w_values.append(w_current)
        V_values.append(V_current)
        t_values.append(t_current)

    # Graficar la variable V en función de t
    plt.plot(w_values,V_values)

#plt.plot(t_values, V_values)
plt.xlabel('Voltaje $V$ (mV)')
plt.ylabel('Varible inhibitoria $w$ ()')
plt.title('Modelo Fitz-Nagumo')
plt.show()