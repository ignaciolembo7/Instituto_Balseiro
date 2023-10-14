import numpy as np
import matplotlib.pyplot as plt

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

# Datos de entrada y salida para XOR generalizado 
import itertools
def generate_input(length):
    # Genera todas las combinaciones posibles de 1 y -1 con la longitud deseada
    combinations = list(itertools.product([1, -1], repeat=length))
    x = np.array(combinations)
    y = np.prod(x, axis=1)
    return x, y
N = 5
x_train, y_train = generate_input(N)

#Arquitectura de la red
input_size = N
hidden_sizes = [1,3,5,7,9,11]
output_size = 1
learning_rate = 0.001
num_epochs = 10000
epochs = [i for i in range(0, num_epochs)]
num_exps = 10
num_test = len(x_train)

fig1, ax1 = plt.subplots(figsize=(8,6)) 
fig2, ax2 = plt.subplots(figsize=(8,6)) 
n=1
for hidden_size in hidden_sizes:
    print("Experimento número: ", n)
    loss = np.zeros(num_epochs)
    accs = np.zeros(num_epochs)

    # Pesos iniciales aleatorios
    w = np.random.uniform(-1,1,size=(hidden_size, input_size)) #filas #columnas
    W = np.random.uniform(-1,1,size=(output_size, hidden_size)) 
    b1 = np.random.uniform(-1,1,size=(1,hidden_size)).reshape(-1, 1) 
    b2 = np.random.uniform(-1,1,size=(1,output_size)).reshape(-1, 1)

    for epoch in epochs:
        #print("epoch",epoch)
        acc = 0  
        for u in range(num_test):

            x = x_train[u].reshape(-1, 1)
            y = y_train[u].reshape(-1, 1)

            # Forward propagation
            h1 = np.dot(w, x) + b1 #input de la capa oculta
            V = act(h1) #output de la capa oculta
            h2 = np.dot(W, V) + b2 #input de la capa de salida
            O = act(h2) #output de la capa de salida

            # Cálculo de la pérdida
            loss[epoch] += mse_loss(y, O)
        
            #Calculo de la precision 
  
            if(abs(O - y) < 0.1*abs(y)):
                acc += 1
            accs[epoch] = acc/num_test
                
            # Backpropagation
            delta2 = dact(h2)*(y-O)
            delta1 = dact(h1)*np.dot(W.T,delta2)

            # Actualización de pesos
            W += learning_rate*np.dot(delta2, V.T) 
            b2 += learning_rate*delta2*(1) 
            w += learning_rate*np.dot(delta1, x.T) 
            b1 += learning_rate*delta1*(1) 

            #if epoch == num_epochs-1:
                #print("Output ", O)
                #print("W ", W)
                #print("w ", w)

        if epoch % 500 == 0:
            print(f'Epoch {epoch}: Loss = {loss[epoch]}')

    n += 1

    print("Entrenamiento completado.")

    ax1.semilogx(epochs, loss, "-", label = "N = " + str(hidden_size))
    ax1.set_xlabel(r"Época", fontsize=18)
    ax1.set_ylabel(r"Error cuádratico medio", fontsize=18)
    ax1.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax1.tick_params(axis='x', rotation=0, labelsize=18, color='black')
    ax1.tick_params(axis='y', labelsize=18, color='black')
    ax1.legend(fontsize=12, framealpha=1, loc = "lower left")
    #ax1.set_xlim(0, 10000)
    ax1.text(0.05, 0.95, r'A' , transform=ax1.transAxes, fontsize=24, verticalalignment='top', fontweight='bold', color="black")
    ax1.grid(True, linewidth=0.5, linestyle='-', alpha=0.9)
    fig1.savefig(f"../Redes-Neuronales/Practica_4/resultados/ej2/mse_ej2.pdf")
    fig1.savefig(f"../Redes-Neuronales/Practica_4/resultados/ej2/mse_ej2.png", dpi=600)

    ax2.semilogx(epochs, accs, "-", label = "N = " + str(hidden_size))
    ax2.set_xlabel(r"Época", fontsize=18)
    ax2.set_ylabel(r"Precisión", fontsize=18)
    ax2.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax2.tick_params(axis='x', rotation=0, labelsize=18, color='black')
    ax2.tick_params(axis='y', labelsize=18, color='black')
    ax2.legend(fontsize=12, framealpha=1, loc = "center left")
    ax2.text(0.05, 0.95, r'B' , transform=ax2.transAxes, fontsize=24, verticalalignment='top', fontweight='bold', color="black")
    #ax2.set_xlim(0, 10000)
    ax2.grid(True, linewidth=0.5, linestyle='-', alpha=0.9)
    fig2.savefig(f"../Redes-Neuronales/Practica_4/resultados/ej2/acc_ej2.pdf")
    fig2.savefig(f"../Redes-Neuronales/Practica_4/resultados/ej2/acc_ej2.png", dpi=600)
