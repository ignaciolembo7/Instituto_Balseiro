# Ejercicio 1: Red Neuronal Feed Forward from Scratch

El objetivo de este ejercicio es implementar una red neuronal para la clasificación de imágenes del conjunto de datos CIFAR-10 desde cero utilizando únicamente la librería NumPy (solamente se usa tensorflow para cargar el dataset por simplicidad). Para ello debes completar el script `ej1.py`.

## Descripción del Problema
CIFAR-10 es un conjunto de datos que contiene 60,000 imágenes (50,000 para train y 10,000 para test) en color (RGB) de 32x32 píxeles, distribuidas en 10 clases diferentes (como aviones, automóviles, aves, etc.). Para este ejercicio, implementarás una red neuronal con la siguiente arquitectura:

Una capa de entrada que recibe las imágenes aplanadas (tamaño 32x32x3).
Una capa oculta con 100 neuronas y función de activación ReLU.
Una capa de salida con 10 neuronas y función de activación softmax, una por cada clase.

¡Suerte joven padawan!