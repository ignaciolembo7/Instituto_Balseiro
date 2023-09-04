import numpy as np
import matplotlib.pyplot as plt

def goldman(A_in, A_out, V, Da, na, k, T, e):
    u = np.exp(na*e*V/(k*T))
    return ((Da*na*e*V)/(1-u))*(A_out -A_in*u )

A_in_na = 10 #mM 
A_out_na = 145 
A_in_k = 140 
A_out_k = 5 
A_in_cl = 4 
A_out_cl = 140 

na_na = 1
na_k = 1
na_cl = -1
k = 1.380649e-23 # J/K	 
e = 1.6022e-19 #C
T = 3000 #K
Da = 0.1

V = np.linspace(-1, 1, 100)
j_na = goldman(A_in_na, A_out_na, V, Da, na_na, k, T, e)
j_k = goldman(A_in_k, A_out_k, V, Da, na_na, k, T, e)
j_cl = goldman(A_in_cl, A_out_cl, V, Da, na_na, k, T, e)

plt.plot(V, j_na, label = "Na")
plt.plot(V, j_k, label = "K")
plt.plot(V, j_cl, label = "Cl")
plt.legend()
plt.xlabel('Voltaje $V$ (mV)')
plt.ylabel('Corriente $j$ (mA)')
plt.title('Ecuación Goldman-Hogdkin-Katz')
plt.grid(True)
plt.show()