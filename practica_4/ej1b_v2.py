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

# Datos de entrada y salida para XOR
x_train = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])  # Entradas
y_train = np.array([[-1], [1], [1], [-1]])  # Salidas

#Arquitectura de la red
input_size = 2
hidden_size = 1
output_size = 1
learning_rate = 0.05
num_epochs = 10000
epochs = [i for i in range(0, num_epochs)]
num_exps = 10
num_test = len(x_train)

fig1, ax1 = plt.subplots(figsize=(8,6)) 
fig2, ax2 = plt.subplots(figsize=(8,6)) 
fig3, ax3 = plt.subplots(figsize=(8,6)) 
fig4, ax4 = plt.subplots(figsize=(8,6)) 

loss_prom = np.zeros(num_epochs)
accs_prom = np.zeros(num_epochs)
epochs_prom = []
epoch_aux = 0

for n in range(num_exps):
    print("Experimento número: ", n)
    loss = np.zeros(num_epochs)
    accs = np.zeros(num_epochs)

    # Pesos iniciales aleatorios
    w = np.random.uniform(-1,1,size=(hidden_size, input_size)) #filas #columnas
    W = np.random.uniform(-1,1,size=(output_size, hidden_size + input_size))
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
            concatenate = np.vstack((V, x)) 
            h2 = np.dot(W,concatenate) + b2 #input de la capa de salida
            O = act(h2) #output de la capa de salida

            # Cálculo de la pérdida
            loss[epoch] += mse_loss(y, O)
            loss_prom[epoch] += mse_loss(y, O)

            #Calculo de la precision 
  
            if(abs(O - y) < 0.1*abs(y)):
                acc += 1

            # Backpropagation
            delta2 = dact(h2)*(y-O)
            delta1 = dact(h1)*np.dot(W[:, :hidden_size].T,delta2)

            # Actualización de pesos
            W += learning_rate*np.dot(delta2, concatenate.T) 
            b2 += learning_rate*delta2*(1) 
            w += learning_rate*np.dot(delta1, x.T) 
            b1 += learning_rate*delta1*(1) 

            #if epoch == num_epochs-1:
                #print("Output ", O)
                #print("W ", W)
                #print("w ", w)
    
        accs[epoch] += acc/num_test
        accs_prom[epoch] += acc/num_test

        #if epoch % 500 == 0:
        #    print(f'Epoch {epoch}: Loss = {loss[epoch]}')
    
    found_epoch = False
    for epoch in range(num_epochs-1, -1, -1):
        if (loss[epoch] > 0.1 and not found_epoch):
            epoch_aux = epoch
            found_epoch = True
        if found_epoch:
            break
    if(loss[-1] < 0.1):
        epochs_prom.append(epoch_aux)
    
    print("Entrenamiento " + str(n) + " completado.")

    ax1.semilogx(epochs, loss, "-")
    ax1.set_xlabel(r"Época", fontsize=18)
    ax1.set_ylabel(r"Error cuádratico medio", fontsize=18)
    ax1.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax1.tick_params(axis='x', rotation=0, labelsize=18, color='black')
    ax1.tick_params(axis='y', labelsize=18, color='black')
    #ax1.set_xlim(0, 10000)
    ax1.text(0.05, 0.95, r'C' , transform=ax1.transAxes, fontsize=24, verticalalignment='top', fontweight='bold', color="black")
    ax1.grid(True, linewidth=0.5, linestyle='-', alpha=0.9)
    fig1.savefig(f"../Redes-Neuronales/Practica_4/resultados/ej1b/mse_ej1b.pdf")
    fig1.savefig(f"../Redes-Neuronales/Practica_4/resultados/ej1b/mse_ej1b.png", dpi=600)

    ax2.semilogx(epochs, accs, "-")
    ax2.set_xlabel(r"Época", fontsize=18)
    ax2.set_ylabel(r"Precisión", fontsize=18)
    ax2.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax2.tick_params(axis='x', rotation=0, labelsize=18, color='black')
    ax2.tick_params(axis='y', labelsize=18, color='black')
    #ax2.set_xlim(0, 10000)
    ax2.text(0.05, 0.95, r'D' , transform=ax2.transAxes, fontsize=24, verticalalignment='top', fontweight='bold', color="black")
    ax2.grid(True, linewidth=0.5, linestyle='-', alpha=0.9)
    fig2.savefig(f"../Redes-Neuronales/Practica_4/resultados/ej1b/acc_ej1b.pdf")
    fig2.savefig(f"../Redes-Neuronales/Practica_4/resultados/ej1b/acc_ej1b.png", dpi=600)

loss_prom = loss_prom/num_exps
accs_prom = accs_prom/num_exps

print(epochs_prom)
print("Epoca de convergencia", np.mean(epochs_prom))

if len(epochs) == len(loss_prom) == len(accs_prom):
    # Combina los arrays en una sola matriz
    data = np.column_stack((epochs, loss_prom, accs_prom))

    # Especifica el nombre de archivo en el que deseas guardar los datos
    file_name = "../Redes-Neuronales/Practica_4/resultados/ej1b/prom_ej1b.txt"

    # Guarda los datos en el archivo de texto
    np.savetxt(file_name, data, fmt="%d %.6f %.6f")