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
    E= (y - O) ** 2
    return 0.5*np.sum(np.sum(E, axis=1))

#dudas
#en cada epoca los pesos son los mismos para cada test?    


# Datos de entrada y salida para XOR
x = np.array([(0, 0), [0, 1], [1, 0], [1, 1]])  # Entradas
y = np.array([[0], [1], [1], [0]])  # Salidas
bias1 = np.array([[-1,-1], [-1,-1], [-1,-1], [-1,-1]]) 
bias2 = np.array([[-1], [-1], [-1], [-1]]) 

#Arquitectura de la red
input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.1
num_epochs = 1
num_exps = 1

m = 1 #cantidad de capas
fig1, ax1 = plt.subplots(figsize=(8,6)) 
fig2, ax2 = plt.subplots(figsize=(8,6)) 

for n in range(num_exps):

    # Pesos iniciales aleatorios
    w = np.random.uniform(size=(input_size, hidden_size)) #filas #columnas
    W = np.random.uniform(size=(hidden_size, output_size))
    b1 = np.random.uniform(size=(1,hidden_size))
    b2 = np.random.uniform(size=(1,output_size))


    epochs = [i for i in range(0, num_epochs)]
    loss = np.zeros(num_epochs)
    accs = np.zeros(num_epochs)

    for epoch in epochs:
        print("epoch",epoch)
        # Forward propagation

        print("x",x.T)
        print("x",x)

        h1 = np.dot(x, w) + np.dot(bias1, b1.T) #input de la capa oculta
        V = act(h1) #output de la capa oculta
        h2 = np.dot(V, W) + np.dot(bias2, b2.T) #input de la capa de salida
        O = act(h2) #output de la capa de salida
        # Cálculo de la pérdida
        loss[epoch] = mse_loss(y, O)

        # Calculo de la precision 
        acc = 0
        for j in range(len(O)):
            if(O[j] - y[j] < 0.1*y[j]):
                acc += 1
        accs[epoch] = acc/len(O)
            
        # Backpropagation
        delta2 = dact(h2)*(y-O)
        delta1 = dact(h1)*np.dot(delta2,W.T)
        print("delta2",delta2)
        print("delta1",delta1)
        # Actualización de pesos
        W += learning_rate*np.dot(V.T,delta2) 
        b2 += learning_rate*np.dot(V.T,delta2) 
        w += learning_rate*np.dot(x.T,delta1) 
        b1 += learning_rate*np.dot(x.T,delta1) 
        
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}: Loss = {loss}')

    print("Entrenamiento completado.")

    # Evaluación del modelo
    hidden_layer = act(np.dot(x, w))
    output_layer = act(np.dot(V, W))

    ax1.plot(epochs, loss, "-")
    ax1.set_xlabel(r"Época", fontsize=18)
    ax1.set_ylabel(r"Error cuádratico medio", fontsize=18)
    ax1.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax1.tick_params(axis='x', rotation=0, labelsize=18, color='black')
    ax1.tick_params(axis='y', labelsize=18, color='black')
    ax1.set_xlim(0, 10000)
    ax1.grid(True, linewidth=0.5, linestyle='-', alpha=0.9)
    fig1.savefig(f"../Redes-Neuronales/Practica_4/resultados/mse_1a.pdf")
    fig1.savefig(f"../Redes-Neuronales/Practica_4/resultados/mse_1a.png", dpi=600)

    ax2.plot(epochs, accs, "-")
    ax2.set_xlabel(r"Época", fontsize=18)
    ax2.set_ylabel(r"Precisión", fontsize=18)
    ax2.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax2.tick_params(axis='x', rotation=0, labelsize=18, color='black')
    ax2.tick_params(axis='y', labelsize=18, color='black')
    ax2.set_xlim(0, 10000)
    ax2.grid(True, linewidth=0.5, linestyle='-', alpha=0.9)
    fig2.savefig(f"../Redes-Neuronales/Practica_4/resultados/acc_1a.pdf")
    fig2.savefig(f"../Redes-Neuronales/Practica_4/resultados/acc_1a.png", dpi=600)


    print("Predicciones finales:")
    print(output_layer)