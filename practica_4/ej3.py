import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def logistic_map(n):
    x = np.zeros(n+1)
    x[0] = np.random.random()
    for i in range(n):
        x[i+1] = 4 * x[i] * (1 - x[i])
    y = x[1:]
    x = x[:-1]
    return x, y

ntrains = [5,10,100]
ntest = 100
epochs = 3000
learning_rate = 0.01
colors = ['b','orange','g']
c = 0
fig1, ax1 = plt.subplots(figsize=(8,6)) 

for ntrain in ntrains:

    seed=2                            # for reproducibility 
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Network architecture
    input_dim = 1
    hidden_dim = 5    # Numero de unidades ocultas
    output_dim = 1

    inputs = tf.keras.layers.Input(shape=(input_dim,))
    hidden = tf.keras.layers.Dense(hidden_dim, activation='sigmoid')(inputs)
    merge=tf.keras.layers.concatenate([inputs,hidden])
    output = tf.keras.layers.Dense(1, activation='linear')(merge)

    # Data Input
    x_train=np.zeros(ntrain, dtype=np.float32)
    y_train=np.zeros(ntrain, dtype=np.float32)
    x_train, y_train = logistic_map(ntrain)
        
    x_test = np.zeros(ntest, dtype=np.float32)
    y_test = np.zeros(ntest, dtype=np.float32)
    x_test, y_test = logistic_map(ntest)

    # Model 
    
    opti=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, decay=0.0)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=opti,loss='MSE')
    print("Entrenamiento con " + str(ntrain) + " ejemplos.")
    history=model.fit(x=x_train, y=y_train,
                    epochs=epochs,
                    #batch_size=4,
                    shuffle=False,
                    validation_data=(x_test, y_test), 
                    verbose=True)
    
    prediction = model.predict(x_test, verbose=True)
    
    tf.keras.utils.plot_model(model, to_file='../Redes-Neuronales/Practica_4/resultados/ej3/logmap.png', show_shapes=False, show_layer_names=True, rankdir='TB')
    print(model.summary())

    #####################################################################
    # Output files

    W_Input_Hidden = model.layers[1].get_weights()[0]
    W_Output_Hidden = model.layers[3].get_weights()[0]
    B_Input_Hidden = model.layers[1].get_weights()[1]
    B_Output_Hidden = model.layers[3].get_weights()[1]
    #print(summary)
    print('INPUT-HIDDEN LAYER WEIGHTS:')
    print(W_Input_Hidden)
    print('HIDDEN-OUTPUT LAYER WEIGHTS:')
    print(W_Output_Hidden)

    print('INPUT-HIDDEN LAYER BIAS:')
    print(B_Input_Hidden)
    print('HIDDEN-OUTPUT LAYER BIAS:')
    print(B_Output_Hidden)

    # "Loss"
    ax1.semilogy(np.sqrt(history.history['loss']),'-', color = colors[c],  label = "Entrenamiento - " + str(ntrain) + " ejemplos")
    ax1.semilogy(np.sqrt(history.history['val_loss']), '--', color = colors[c], label = "Validación - " + str(ntrain) + " ejemplos")
    #ax1.semilogy(history.history['v1_accuracy'], '-.', color = colors[c], label = "v1_accuracy - " + str(ntrain) + " ejemplos")
    #ax1.set_title('model loss')
    ax1.set_xlabel(r"Época", fontsize=18)
    ax1.set_ylabel(r"MSE", fontsize=18)
    ax1.legend(fontsize=12, framealpha=1)
    ax1.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax1.tick_params(axis='x',rotation=0, labelsize=18, color='black')
    ax1.tick_params(axis='y', labelsize=18, color='black')
    #ax1.set_xlim(-100, epochs)
    ax1.grid(True, linewidth=0.5, linestyle='-', alpha=0.9)
    fig1.savefig(f"../Redes-Neuronales/Practica_4/resultados/ej3/ej3_loss.pdf")
    fig1.savefig(f"../Redes-Neuronales/Practica_4/resultados/ej3/ej3_loss.png", dpi=600)

    #Ploteo del mapeo logistico
    fig2, ax2 = plt.subplots(figsize=(8,6)) 
    ax2.plot(x_test, y_test, "o", label = 'Conjunto de validación')
    ax2.plot(x_test, prediction, "o", label = 'Predicción')
    ax2.plot(x_train, y_train, "o", label = 'Conjunto de entrenamiento')
    ax2.set_xlabel(r"$x_n$", fontsize=18)
    ax2.set_ylabel(r"$x_{n+1}$", fontsize=18)
    ax2.legend(title='Entrenamiento con ' + str(ntrain) + ' ejemplos.', fontsize=15, framealpha=1, loc='lower center')
    ax2.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax2.tick_params(axis='x',rotation=0, labelsize=18, color='black')
    ax2.tick_params(axis='y', labelsize=18, color='black')
    #ax2.set_xlim(0, 1)
    ax2.grid(True, linewidth=0.5, linestyle='-', alpha=0.9)
    fig2.savefig(f"../Redes-Neuronales/Practica_4/resultados/ej3/ej3_logmap_" + str(ntrain) + "_tests.pdf")
    fig2.savefig(f"../Redes-Neuronales/Practica_4/resultados/ej3/ej3_logmap_" + str(ntrain) + "_tests.png", dpi=600)

    c += 1

    #plt.show()