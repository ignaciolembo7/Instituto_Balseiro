import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#Ploteo 
import seaborn as sns
import matplotlib.patches as patches
#sns.axes_style("whitegrid")
sns.set_style("ticks")

def gaussian(i, i_, sigma):
    return np.exp(-(((i-i_)**2)/(2*(sigma**2))))

def input_semicircle():
    f = True
    while f:
        x = np.random.uniform(-1.1,1.1)
        y = np.random.uniform(0,1.1)

        r = np.linalg.norm([x,y])
        if( 0.9 <= r <= 1.1 and 0 <= np.arctan2(y,x) <= np.pi ):
            f = False
    return np.array([x,y])

def input_triangle():
    f = True
    while f:
        x = np.random.uniform(-0.5,0.5)
        y = np.random.uniform(0.0,np.sqrt(3)/2)
                
        if ( -0.5 <= x <= 0.5 and 0 <= y <= np.tan(np.pi/3)*(0.5-np.abs(x)) ):
            f = False
    #ax1.plot(x, y, marker='o', markersize=1, color='k')     
    return np.array([x,y])

def input_square():
    f = True
    while f:
        x = np.random.uniform(-0.5,0.5)
        y = np.random.uniform(0.0,2.0)
        # Verifica si el punto está en la mitad inferior del cuadrado
        if (-0.5 <= x <= 0.5) and (0.0 <= y <= 1.0):
            # Genera un número aleatorio adicional para duplicar la probabilidad en la mitad inferior
            if y <= 0.5:
                if np.random.rand() <= 0.5:
                    f = False
            else:
                f = False
    #ax1.plot(x, y, marker='o', markersize=0.5, color='k')  
    return np.array([x,y])

#Arquitectura de la red
input_size = 2
output_size = 10
learning_rate = 0.01
sigma_0 = 10000
#sigma = 1.0
num_epochs = 100000
fig_index = "C"
epochs = [i for i in range(0, num_epochs)]

w_plot = np.zeros((num_epochs, input_size))

w = np.array([[np.random.uniform(-0.75, 0.75), np.random.uniform(-0.25, 1.25)] for _ in range(output_size)])

"""
w = np.array([[ 0.1389785 ,  0.02162108],
 [ 0.00636808 ,  0.16040874],
 [ 0.05046116 , 0.0006476 ],
 [ 0.15113858 , 0.04233544],
 [ 0.08499907 , 0.13616288],
 [-0.09924555 , 0.15332305],
 [-0.00520648 ,  0.08107896],
 [-0.18375063 , 0.15545262],
 [-0.03089037 , 0.15672824],
 [-0.06498289 ,  0.07104595]])
"""

plot_w = np.zeros((num_epochs, output_size, input_size))

fig1, ax1 = plt.subplots()

for i in range (output_size):
    ax1.plot(w[i, 0], w[i, 1], 'o',  color='k', markersize = 3)

for epoch in tqdm(range(num_epochs)):
    x = input_square()
    norm = np.linalg.norm(w - x, axis=1)
    i_ = np.argmin(norm)
    sigma = sigma_0/(epoch+1)
    # Actualización de pesos
    for i in range(output_size):
        w[i] += learning_rate*gaussian(i,i_,sigma)*(x-w[i])
        plot_w[epoch][i] = w[i]

for i in range ( output_size ):
    ax1.text(w[i, 0], w[i, 1], str(i), fontsize=15, ha='center', va='center', color='k')


for i in range(output_size):
    x_trajectory = plot_w[:, i, 0]
    y_trajectory = plot_w[:, i, 1]
    plt.plot(x_trajectory, y_trajectory, label=f"Neurona {i+1}")

print("Entrenamiento completado.")

ax1.set_xlabel(r"$x$", fontsize=18)
ax1.set_ylabel(r"$y$", fontsize=18)
ax1.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
ax1.tick_params(axis='x', rotation=0, labelsize=18, color='black')
ax1.tick_params(axis='y', labelsize=18, color='black')
#ax1.legend(fontsize=12, framealpha=1)
#ax1.set_xlim(-1.2, 1.2)
#ax1.set_ylim(-0.1, 1.2)
ax1.text(0.05, 0.95, f'{fig_index}' , transform=ax1.transAxes, fontsize=24, verticalalignment='top', fontweight='bold', color="black")
ax1.grid(True, linewidth=0.5, linestyle='-', alpha=0.9)
ax1.set_aspect('equal')

#ax1.plot(0, 0.6387, marker='x', markersize=10, color='k', label='Centro de masa')
ring = patches.Wedge(center=(0, 0), r=1.1, theta1=0, theta2=180, width=0.2, facecolor='lightgray', edgecolor='black', alpha=0.5)
ax1.add_patch(ring)
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-0.5, 1.5)

#triangle = patches.Polygon(np.array([[-0.5, 0], [0.5, 0], [0, np.sqrt(3) / 2]]), closed=True, facecolor='lightgray', edgecolor='black')
#ax1.add_patch(triangle)
#ax1.plot(0, 0.28867, marker='x', markersize=15, color='k', label='Centro de masa')
#ax1.set_xlim(-0.75, 0.75)
#ax1.set_ylim(-0.75, 1.25)

#square = patches.Rectangle((-0.5, 0), 1, 1, angle=0, facecolor='lightgray', edgecolor='black')
#ax1.add_patch(square)
#ax1.plot(0, 0.583, marker='x', markersize=15, color='k', label='Centro de masa')
#ax1.set_xlim(-0.75, 0.75)
#ax1.set_ylim(-0.25, 1.25)

fig1.savefig(f"../Redes-Neuronales/Practica_5/resultados/ej2/pesos_sigma_{sigma}_epochs_{num_epochs}_lr_{learning_rate}.pdf")
fig1.savefig(f"../Redes-Neuronales/Practica_5/resultados/ej2/pesos_sigma_{sigma}_epochs_{num_epochs}_lr_{learning_rate}.png", dpi=600)