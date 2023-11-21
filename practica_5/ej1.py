import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#Ploteo 
import seaborn as sns
#sns.axes_style("whitegrid")
sns.set_style("ticks")


Sigma = np.array([
    [2,1,1,1],
    [1,2,1,1],
    [1,1,2,1],
    [1,1,1,2]
])

sqrtSigma = np.array([
    [1.309, 0.309, 0.309, 0.309],
    [0.309, 1.309, 0.309, 0.309],
    [0.309, 0.309, 1.309, 0.309],
    [0.309, 0.309, 0.309, 1.309]
])

#Arquitectura de la red
input_size = 4
output_size = 1
learning_rate = 0.01
num_epochs = 250
fig_index = "A"
epochs = [i for i in range(0, num_epochs)]

w_plot = np.zeros((num_epochs, input_size))
dot_plot = np.zeros((num_epochs, input_size))

# Pesos iniciales aleatorios
w = np.random.uniform(-0.1,0.1,size=(output_size, input_size)) #filas #columnas
print(w)
autovalores, autovectores = np.linalg.eig(Sigma)

print("Autovalores de Sigma:")
print(autovalores)
print("Autovectores de Sigma:")
print((autovectores.T[1]).reshape(-1, 1))
print("\n")

for epoch in tqdm(range(num_epochs)):

    w_plot[epoch] = w
    z_train = np.random.multivariate_normal([0,0,0,0], Sigma)
    x_train = np.dot(sqrtSigma,z_train)

    O = np.dot(w, x_train)

    for i in range(input_size):
        print(w)
        print(autovectores.T[i].reshape(-1, 1))
        dot_plot[epoch][i] = np.dot(w, autovectores.T[i].reshape(-1, 1))
    
    # Actualización de pesos
    w += learning_rate*O*(x_train-O*w)

print("Entrenamiento completado.")

fig1, ax1 = plt.subplots(figsize=(8,6))
for i in range(input_size):
    ax1.plot(epochs, w_plot[:,i], "-", label = r"$w_{" + str(i) + "}$")
ax1.hlines(y=0.5,  xmin=0, xmax=num_epochs, linestyle='--', color='gray')
ax1.hlines(y=-0.5,  xmin=0, xmax=num_epochs, linestyle='--', color='gray')
ax1.set_xlabel(r"Época", fontsize=18)
ax1.set_ylabel(r"Pesos $w_j$", fontsize=18)
ax1.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
ax1.tick_params(axis='x', rotation=0, labelsize=18, color='black')
ax1.tick_params(axis='y', labelsize=18, color='black')
ax1.legend(fontsize=12, framealpha=1, loc= "center right")
ax1.set_xlim(0, num_epochs)
ax1.text(0.05, 0.9, f'{fig_index}' , transform=ax1.transAxes, fontsize=24, verticalalignment='top', fontweight='bold', color="black")
ax1.grid(True, linewidth=0.5, linestyle='-', alpha=0.9)
#fig1.savefig(f"../Redes-Neuronales/Practica_5/resultados/ej1/pesos_lr_{learning_rate}.pdf")
#fig1.savefig(f"../Redes-Neuronales/Practica_5/resultados/ej1/pesos_lr_{learning_rate}.png", dpi=600)

fig2, ax2 = plt.subplots(figsize=(8,6))
for i in range(input_size):
    ax2.plot(epochs, dot_plot[:, i], "-", label=r"$\vec{w} \cdot \vec{v_" + str(i) + "}$")
ax2.hlines(y=1.0,  xmin=0, xmax=num_epochs, linestyle='--', color='gray')
ax2.hlines(y=-1.0,  xmin=0, xmax=num_epochs, linestyle='--', color='gray')
ax2.hlines(y=0.0,  xmin=0, xmax=num_epochs, linestyle='--', color='gray')
ax2.set_xlabel(r"Época", fontsize=18)
ax2.set_ylabel(r"Producto escalar $\vec{w} \cdot \vec{v_j}$", fontsize=18)
ax2.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
ax2.tick_params(axis='x', rotation=0, labelsize=18, color='black')
ax2.tick_params(axis='y', labelsize=18, color='black')
ax2.legend(fontsize=12, framealpha=1, loc= "upper right")
ax2.set_xlim(0, num_epochs)
ax2.text(0.05, 0.9, f'{fig_index}', transform=ax2.transAxes, fontsize=24, verticalalignment='top', fontweight='bold', color="black")
ax2.grid(True, linewidth=0.5, linestyle='-', alpha=0.9)
#fig2.savefig(f"../Redes-Neuronales/Practica_5/resultados/ej1/prodesc_lr_{learning_rate}.pdf")
#fig2.savefig(f"../Redes-Neuronales/Practica_5/resultados/ej1/prodesc_lr_{learning_rate}.png", dpi=600)