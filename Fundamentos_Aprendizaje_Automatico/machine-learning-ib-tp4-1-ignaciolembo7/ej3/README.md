# (Optativo) Ejercicio 3: Micro Keras
El objetivo de este ejercicio es desarrollar una biblioteca modular en Python para la construcción de redes neuronales feedforward de manera flexible desde cero, utilizando solo la librería NumPy. 

## Estructura del Código
La biblioteca deberá estar organizada organizada en los siguientes módulos:

1. metrics.py
2. losses.py
3. activations.py
4. models.py
5. layers.py
6. optimizers.py

## Descripción de los Módulos
### 1. metrics.py

Accuracy: Calcula la precisión de las predicciones.
MSE: Calcula el error cuadrático medio (Mean Squared Error).

### 2. losses.py

Loss: Interfaz de las funciones de costo que define el método __call__ y gradient.
MSE: Implementación de la función de costo Mean Squared Error.

### 3. activations.py

ReLU: Implementa la función de activación ReLU y su derivada.
Tanh: Implementa la función de activación Tanh y su derivada.
Sigmoid: Implementa la función de activación Sigmoid y su derivada.

### 4. models.py

Network: Clase que implementa una red neuronal feedforward. Deberá permitir agregar capas, compilar el modelo, realizar forward propagation, backward propagation, entrenar el modelo y hacer predicciones.

### 5. layers.py

BaseLayer: Clase base para cualquier tipo de capa. Define las interfaces forward y backward.
Input: Representa la capa de entrada de la red neuronal, heredando de BaseLayer.
Layer: Clase base para capas con pesos. Hereda de BaseLayer.
Dense: Representa una capa densa (fully connected) que hereda de Layer.

### 6. optimizers.py

Optimizer: Interfaz para optimizadores. Define el método update.
SGD: Implementa el optimizador Stochastic Gradient Descent.

## Caso de prueba
Para validar la implementación, se utilizará el problema XOR (un problema facilito para que puedan hacer pruebas rápidas).