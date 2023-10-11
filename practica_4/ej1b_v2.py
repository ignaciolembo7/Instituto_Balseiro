import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.optimize import curve_fit

#Ploteo 
import seaborn as sns
#sns.axes_style("whitegrid")
sns.set_style("ticks")

# Funcion de activacion
def act(x):
    return np.tanh(x)
def dact(x):
    return 1 - np.tanh(x) ** 2
# Funcion de coste (error cuadratico medio)
def mse_loss(y, O):
    E = (y - O) ** 2
    return 0.5*np.sum(E)

# Datos de entrada y salida para XOR
x_train = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])  # Entradas
y_train = np.array([[-1], [1], [1], [-1]])  # Salidas

#Arquitectura de la red
input_size = 2
hidden_size = 1
output_size = 1
learning_rate = 0.01
num_epochs = 10000
epochs = [i for i in range(0, num_epochs)]
num_exps = 10
num_test = len(x_train)

fig1, ax1 = plt.subplots(figsize=(8,6)) 
fig2, ax2 = plt.subplots(figsize=(8,6)) 

for n in range(num_exps):
    print("Experimento número: ", n)
    loss = np.zeros(num_epochs)
    accs = np.zeros(num_epochs)

    # Pesos iniciales aleatorios
    w = np.random.uniform(size=(hidden_size, input_size)) #filas #columnas
    W = np.random.uniform(size=(output_size, hidden_size + input_size))
    b1 = np.random.uniform(size=(1,hidden_size)).reshape(-1, 1) 
    b2 = np.random.uniform(size=(1,output_size)).reshape(-1, 1) 

    for epoch in epochs:
        #print("epoch",epoch)
        acc = 0  
        for u in range(num_test):

            x = x_train[u].reshape(-1, 1)
            y = y_train[u].reshape(-1, 1)

            # Forward propagation
            h1 = np.dot(w, x) + b1 #input de la capa oculta
            V = act(h1) #output de la capa oculta
            concatenate = np.vstack((V, x)) 
            h2 = np.dot(W,concatenate) + b2 #input de la capa de salida
            O = act(h2) #output de la capa de salida

            # Cálculo de la pérdida
            loss[epoch] += mse_loss(y, O)
        
            #Calculo de la precision 
  
            if(abs(O - y) < 0.1*abs(y)):
                acc += 1
            accs[epoch] = acc/num_test
                
            # Backpropagation
            delta2 = dact(h2)*(y-O)
            delta1 = dact(h1)*np.dot(W[:, :hidden_size].T,delta2)

            # Actualización de pesos
            W += learning_rate*np.dot(delta2, concatenate.T) 
            b2 += learning_rate*delta2*(1) 
            w += learning_rate*np.dot(delta1, x.T) 
            b1 += learning_rate*delta1*(1) 

            if epoch == num_epochs-1:
                print("Output ", O)
                #print("W ", W)
                #print("w ", w)

        if epoch % 500 == 0:
            print(f'Epoch {epoch}: Loss = {loss[epoch]}')


    print("Entrenamiento completado.")

    ax1.plot(epochs, loss, "-")
    ax1.set_xlabel(r"Época", fontsize=18)
    ax1.set_ylabel(r"Error cuádratico medio", fontsize=18)
    ax1.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax1.tick_params(axis='x', rotation=0, labelsize=18, color='black')
    ax1.tick_params(axis='y', labelsize=18, color='black')
    #ax1.set_xlim(0, 10000)
    ax1.grid(True, linewidth=0.5, linestyle='-', alpha=0.9)
    fig1.savefig(f"../Redes-Neuronales/Practica_4/resultados/ej1b/mse_1b.pdf")
    fig1.savefig(f"../Redes-Neuronales/Practica_4/resultados/ej1b/mse_1b.png", dpi=600)

    ax2.plot(epochs, accs, "-")
    ax2.set_xlabel(r"Época", fontsize=18)
    ax2.set_ylabel(r"Precisión", fontsize=18)
    ax2.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax2.tick_params(axis='x', rotation=0, labelsize=18, color='black')
    ax2.tick_params(axis='y', labelsize=18, color='black')
    #ax2.set_xlim(0, 10000)
    ax2.grid(True, linewidth=0.5, linestyle='-', alpha=0.9)
    fig2.savefig(f"../Redes-Neuronales/Practica_4/resultados/ej1b/acc_1b.pdf")
    fig2.savefig(f"../Redes-Neuronales/Practica_4/resultados/ej1b/acc_1b.png", dpi=600)