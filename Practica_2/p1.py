import numpy as np
import matplotlib.pyplot as plt

#Constantes 
gna = 120 #mS/cm2
gk = 36 #mS/cm2
gcl = 0.3 #mS/cm2
Vna = 50 #mV
Vk = -77 #mV
Vcl = -54.4 #mV
C = 1 #muF/cm2
Iext = 10 #muA/cm2
gsyn = 0 
ts = 3 #ms
Vsyn = 0 #mV /-80 mV
tau = 3 #ms

#Parámetros de la simulacion 
t0 = 0
t_tot = 1000 #ms
dt = 0.1

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

# Definición de las ecuaciones diferenciales
def f_n(t, s, n, m, h, V):
    # Define la ecuación para dn/dt
    return (n0(V) - n)/tn(V)   

def f_m(t, s, n, m, h, V):
    # Define la ecuación para dm/dt
    return (m0(V) - m)/tm(V)

def f_h(t, s, n, m, h, V):
    # Define la ecuación para dh/dt
    return (h0(V) - h)/th(V)

def f_V(t, s, n, m, h, V):
    # Define la ecuación para dV/dt
    return (Iext - gk*(n**4)*(V-Vk) - gna*(m**3)*h*(V-Vna) - gcl*(V-Vcl) - gsyn*s*(V-Vsyn))/C  

def f_s(t, s, n , m, h, V, Vpre):
    return (s0(Vpre) - s)/ts 

def runge_kutta(t, s, n, m, h, V, Vpre):

    k1_s = dt * f_s(t, s, n, m, h, V, Vpre)
    k1_n = dt * f_n(t, s, n, m, h, V)
    k1_m = dt * f_m(t, s, n, m, h, V)
    k1_h = dt * f_h(t, s, n, m, h, V)
    k1_V = dt * f_V(t, s, n, m, h, V)
    
    k2_s = dt * f_s(t + dt/2, s + k1_s/2, n + k1_n/2, m + k1_m/2, h + k1_h/2, V + k1_V/2, Vpre)
    k2_n = dt * f_n(t + dt/2, s + k1_s/2, n + k1_n/2, m + k1_m/2, h + k1_h/2, V + k1_V/2)
    k2_m = dt * f_m(t + dt/2, s + k1_s/2, n + k1_n/2, m + k1_m/2, h + k1_h/2, V + k1_V/2)
    k2_h = dt * f_h(t + dt/2, s + k1_s/2, n + k1_n/2, m + k1_m/2, h + k1_h/2, V + k1_V/2)
    k2_V = dt * f_V(t + dt/2, s + k1_s/2, n + k1_n/2, m + k1_m/2, h + k1_h/2, V + k1_V/2)
    
    k3_s = dt * f_s(t + dt/2, s + k2_s/2, n + k2_n/2, m + k2_m/2, h + k2_h/2, V + k2_V/2, Vpre)
    k3_n = dt * f_n(t + dt/2, s + k2_s/2, n + k2_n/2, m + k2_m/2, h + k2_h/2, V + k2_V/2)
    k3_m = dt * f_m(t + dt/2, s + k2_s/2, n + k2_n/2, m + k2_m/2, h + k2_h/2, V + k2_V/2)
    k3_h = dt * f_h(t + dt/2, s + k2_s/2, n + k2_n/2, m + k2_m/2, h + k2_h/2, V + k2_V/2)
    k3_V = dt * f_V(t + dt/2, s + k2_s/2, n + k2_n/2, m + k2_m/2, h + k2_h/2, V + k2_V/2)
    
    k4_s = dt * f_s(t + dt, s + k3_s, n + k3_n, m + k3_m, h + k3_h, V + k3_V, Vpre)
    k4_n = dt * f_n(t + dt, s + k3_s, n + k3_n, m + k3_m, h + k3_h, V + k3_V)
    k4_m = dt * f_m(t + dt, s + k3_s, n + k3_n, m + k3_m, h + k3_h, V + k3_V)
    k4_h = dt * f_h(t + dt, s + k3_s, n + k3_n, m + k3_m, h + k3_h, V + k3_V)
    k4_V = dt * f_V(t + dt, s + k3_s, n + k3_n, m + k3_m, h + k3_h, V + k3_V)
    
    t += dt
    s += (k1_s + 2*k2_s + 2*k3_s + k4_s) / 6
    n += (k1_n + 2*k2_n + 2*k3_n + k4_n) / 6
    m += (k1_m + 2*k2_m + 2*k3_m + k4_m) / 6
    h += (k1_h + 2*k2_h + 2*k3_h + k4_h) / 6
    V += (k1_V + 2*k2_V + 2*k3_V + k4_V) / 6
    
    return t, s, n, m, h, V


t1_values = np.empty(t_tot + 1)
V1_values = np.empty(t_tot + 1)
t2_values = np.empty(t_tot + 1)
V2_values = np.empty(t_tot + 1)

V0 = Vk

n1 = [t0, s0(V0), n0(V0), m0(V0), h0(V0), V0]
n2 = [10, 1, 1, 1, 1, 1]

for i in range(t_tot):

    Vpre1 = n1[5]
    Vpre2 = n2[5] 

    n1[0], n1[1], n1[2], n1[3], n1[4], n1[5] = runge_kutta(n1[0], n1[1], n1[2], n1[3], n1[4], n1[5], Vpre2)
    n2[0], n2[1], n2[2], n2[3], n2[4], n2[5] = runge_kutta(n2[0], n2[1], n2[2], n2[3], n2[4], n2[5], Vpre1)

    t1_values[i+1] = n1[0]
    t2_values[i+1] = n1[0]
    V1_values[i+1] = n2[5]
    V1_values[i+1] = n2[5]


plt.plot(t1_values,V1_values)
plt.plot(t2_values,V2_values)
plt.show()
