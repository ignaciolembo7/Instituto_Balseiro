import numpy as np
import matplotlib.pyplot as plt
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
def s0(V_pre):
    return 0.5*(1+np.tanh(V_pre/5))

def f_s(s, Vpre):
    return (s0(Vpre) - s)/ts
def f_n(n, V):
    return (n0(V) - n)/tn(V)
def f_m(m, V):
    return (m0(V) - m)/tm(V)
def f_h(h, V):
    return (h0(V) - h)/th(V)
def f_V(s, n, m, h, V):
    return (Iext - gk*(n**4)*(V-Vk) - gna*(m**3)*(h)*(V-Vna) - gcl*(V-Vcl) - gsyn*(s)*(V-Vsyn))/C

def runge_kutta_4(funcs, y0, t0, dt):
    k1 = np.array([func(t0, y0) for func in funcs])
    k2 = np.array([func(t0 + dt/2, y0 + dt/2 * k1) for func in funcs])
    k3 = np.array([func(t0 + dt/2, y0 + dt/2 * k2) for func in funcs])
    k4 = np.array([func(t0 + dt, y0 + dt * k3) for func in funcs])
    
    y_new = y0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y_new

# Definición de las ecuaciones diferenciales
funcs = [
    lambda t, y: f_s(y[0], y[9]) ,
    lambda t, y: f_s(y[1], y[8]) ,
    lambda t, y: f_n(y[2], y[8]) ,
    lambda t, y: f_n(y[3], y[9]) ,
    lambda t, y: f_m(y[4], y[8]) ,
    lambda t, y: f_m(y[5], y[9]) ,
    lambda t, y: f_h(y[6], y[8]) ,
    lambda t, y: f_h(y[7], y[9]) ,
    lambda t, y: f_V(y[0], y[2], y[4], y[6], y[8])  ,
    lambda t, y: f_V(y[1], y[3], y[5], y[7], y[9])  ,
]

#Constantes 
gna = 120 #mS/cm2
gk = 36 #mS/cm2
gcl = 0.3 #mS/cm2
Vna = 50 #mV
Vk = -77 #mV
Vcl = -54.4 #mV
C = 1 #muF/cm2
Iext = 50 #muA/cm2
gsyn = 0
ts = 3 #ms
Vsyn = -80 #mV /-80 mV
tau = 3 #ms

# Valores iniciales: y = [s1, s2, n1, n2, m1, m2, h1, h2, V1, V2]
#y_initial = np.array([s0(Vk), s0(Vk), n0(Vk), n0(Vk), m0(Vk), m0(Vk), h0(Vk)-1, h0(Vk)-1, Vk-1, Vk-1])
y_initial = np.array([s0(Vk), 0, n0(Vk), 0, m0(Vk), 0, h0(Vk), 0, Vk, 0])
t_initial = 0.0
t_final = 150
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
    y = runge_kutta_4(funcs, y, t, time_step)

    t_values[i] = t
    V1_values[i] = y[8]
    V2_values[i] = y[9]
    
    t += time_step

#Ploteo
plt.plot(t_values, V1_values, label = "Neurona 1")
plt.plot(t_values, V2_values, label = "Neurona 2")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Voltaje (mV)")
plt.legend()
plt.show()