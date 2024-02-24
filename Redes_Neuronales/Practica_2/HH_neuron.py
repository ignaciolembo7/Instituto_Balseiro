import numpy as np

#Constantes 
gna = 120 #mS/cm2
gk = 36 #mS/cm2
gcl = 0.3 #mS/cm2
Vna = 50 #mV
Vk = -77 #mV
Vcl = -54.4 #mV
C = 1 #muF/cm2
Iext = 10 #muA/cm2
gsyn = 4
ts = 3 #ms
Vsyn = -80 #mV /-80 mV
tau = 3 #ms

# Definición de las ecuaciones diferenciales
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

