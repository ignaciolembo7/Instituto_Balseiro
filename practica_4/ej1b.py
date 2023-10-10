import numpy as numpy
import matplotlib as plt
import tensorflow as tf

#Esta red respecto al numero de parámetros es más sencilla que la del item 1a


seed = 2
np.random.seed(seed)
tf.random.set_seed(seed)

def output_activation(x):
    return tf.math.sinh(x)/tf.math.cosh(x)

#Network architecture
hidden_dim = 1 #number of hidden units

inputs = tf.keras.layers.Input(shape=(2,))
x = tf.keras.layers.Dense(hidden_dim, activation=output_activation)(inputs)
merge = tf.keras.layers.concatenate([inputs,x],axis=-1)
predictions = tf.keras.layers.Dense(1,activation=output_activation)(merge)

#Data input 
ntrain = 4 
x_train = np.zeros((ntrain,1), dtype = np.float32) #ntrain es mu y 1 es j
y_train = np.zeros((ntrain,1), dtype = np.float32) 

x_train[0,0] = 1 
x_train[0,1] = 1 
y_train[1] = 0

x_train[2,0] = 1 
x_train[2,1] = -1 
y_train[2] = 0 

x_train[3,0] = 1 
x_train[3,1] = -1 
y_train[3] = 1

x_test[:]=
y_test[:]=

print(x_test.shape)
print(y_test.shape)

#Accuracy
from tensorflow.python.keras import backend as K 
def v1_accuracy(y_true, y_pred):
    return 

#Model
opti = tf.keras.optimizers.Adam(lr=0.01) #Adam sistema que incluye momento y gradiente, lr es la tasa de aprendizaje

model.compile(optimizers=opti, loss='MSE', metrics=[v1_accuracy])

history = model.fit(x=x_train, y=y_train, 
                    epochs=500, #pasos de aprendizaje
                    batch_size = 4,
                    shuffle = False,
                    validation_data=(x_test,y_test),
                    verbose=True)

tf.keras.utils.plot_model(model, to_file)

#interpretar geometricamente la matriz y la orientacion del hiperplano