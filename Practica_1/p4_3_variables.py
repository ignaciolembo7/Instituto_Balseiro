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

def n0(V):
    an = 0.01*(V+55)/(1-np.exp((-V-55)/10))
    bn = 0.125*np.exp((-V-65)/80)
    return an/(an+bn)
def tn(V):
    an = 0.01*(V+55)/(1-np.exp((-V-55)/10))
    bn = 0.125*np.exp((-V-65)/80)
    return 1/(an+bn)

def m0(V):
    am = 0.1*(V+40)/(1-np.exp((-V-40)/10))
    bm = 4*np.exp((-V-65)/18)
    return am/(am+bm)
def tm(V):
    am = 0.1*(V+40)/(1-np.exp((-V-40)/10))
    bm = 4*np.exp((-V-65)/18)
    return 1/(am+bm)

def h0(V):
    ah = 0.07*np.exp((-V-65)/20)
    bh = 1/(1+np.exp((-V-35)/10))
    return ah/(ah+bh)
def th(V):
    ah = 0.07*np.exp((-V-65)/20)
    bh = 1/(1+np.exp((-V-35)/10))
    return 1/(ah+bh)

# Definición de las ecuaciones diferenciales
def f_n(t, n, h, V):
    # Define la ecuación para dn/dt
    return (n0(V) - n)/tn(V)   

def f_h(t, n, h, V):
    # Define la ecuación para dh/dt
    return (h0(V) - h)/th(V)

def f_V(t, n, h, V, Iext):
    # Define la ecuación para dV/dt
    return (Iext - gk*(n**4)*(V-Vk) - gna*(m0(V)**3)*h*(V-Vna) - gcl*(V-Vcl))/C  

# Implementación del método de Runge-Kutta de cuarto orden
def runge_kutta_4_system(t, n, h, V, i, dt):
    # Calcula los k1, k2, k3 y k4 para cada variable
    k1_n = dt * f_n(t, n, h, V)
    k1_h = dt * f_h(t, n, h, V)
    k1_V = dt * f_V(t, n, h, V, i)
    
    k2_n = dt * f_n(t + dt/2, n + k1_n/2, h + k1_h/2, V + k1_V/2)
    k2_h = dt * f_h(t + dt/2, n + k1_n/2, h + k1_h/2, V + k1_V/2)
    k2_V = dt * f_V(t + dt/2, n + k1_n/2, h + k1_h/2, V + k1_V/2, i)
    
    k3_n = dt * f_n(t + dt/2, n + k2_n/2, h + k2_h/2, V + k2_V/2)
    k3_h = dt * f_h(t + dt/2, n + k2_n/2, h + k2_h/2, V + k2_V/2)
    k3_V = dt * f_V(t + dt/2, n + k2_n/2, h + k2_h/2, V + k2_V/2, i)
    
    k4_n = dt * f_n(t + dt, n + k3_n, h + k3_h, V + k3_V)
    k4_h = dt * f_h(t + dt, n + k3_n, h + k3_h, V + k3_V)
    k4_V = dt * f_V(t + dt, n + k3_n, h + k3_h, V + k3_V, i)
    
    # Calcula las nuevas variables utilizando los k1, k2, k3 y k4
    new_n = n + (k1_n + 2*k2_n + 2*k3_n + k4_n) / 6
    new_h = h + (k1_h + 2*k2_h + 2*k3_h + k4_h) / 6
    #new_h = n0(Vk) + h0(Vk) - new_n
    new_V = V + (k1_V + 2*k2_V + 2*k3_V + k4_V) / 6
    
    return new_n, new_h, new_V

f = []
I_ext = list(range(1, 30))  # Crear un vector del 1 al 20

for i in I_ext:
    # Parámetros de la simulación
    dt = 0.01  # Tamaño del paso de tiempo
    t_end = 50.0  # Tiempo final de simulación

    # Listas para almacenar resultados
    V0=Vk
    n_values = [n0(V0)]
    h_values = [h0(V0)]
    V_values = [V0]
    t_values = [0]

    # Bucle de integración
    t_current = 0
    n_current = n0(V0)
    h_current = h0(V0)
    V_current = V0

    t_vec = []
    while t_current < t_end:
        n_current, h_current, V_current = runge_kutta_4_system(t_current, n_current, h_current, V_current, i, dt)
        t_current += dt
        
        n_values.append(n_current)
        h_values.append(h_current)
        V_values.append(V_current)
        t_values.append(t_current)

    # Encontrar los picos en la curva
    peaks, _ = find_peaks(V_values, height=0, distance=10)  # Ajusta la prominencia según tu necesidad

    if(len(peaks) > 2):
        f.append(1/(t_values[peaks[-1]] - t_values[peaks[-2]]))
    else:
        f.append(0)

# Graficar la variable V en función de t
    if(i % 30 == 0 ):
        plt.plot(t_values, V_values)
        plt.xlabel('Tiempo $t$ (ms)')
        plt.ylabel('Voltaje $V$ (mV)')
        plt.title('Modelo Hogdkin-Huxley')
                
        for p in peaks:
            #plt.plot(t_values[peaks[-1]], V_values[peaks[-1]], "bo", label="Picos")
            #plt.plot(t_values[peaks[-2]], V_values[peaks[-2]], "ro", label="Picos")
            plt.plot(t_values[p], V_values[p], "ro", label="Picos")
        
        plt.show()


# Graficar la variable V en función de t
plt.plot(I_ext, f, "o")
#plt.plot(t_values, V_values)
plt.xlabel('Corriente $I_{ext}$ (mA)')
plt.ylabel('Frecuencia $f$ ()')
plt.title('Modelo Hogdkin-Huxley')
plt.show()