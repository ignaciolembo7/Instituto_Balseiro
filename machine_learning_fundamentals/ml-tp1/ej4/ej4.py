import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np

def fix(cadena_numeros):
    # Eliminar los corchetes y dividir por espacios para obtener una lista de números como cadenas
    lista_numeros = cadena_numeros.replace("[", "").replace("]", "").replace("\r", "").replace("\n","").split(' ')
    # Filtrar solo los elementos que son números
    lista_numeros = [x for x in lista_numeros if x.replace('.', '', 1).isdigit()]  # Esto filtra los números con puntos decimales
    # Convertir la lista de cadenas en un array de NumPy de tipo float
    array_numeros = np.array(lista_numeros, dtype=float)
    # Imprimir el array de NumPy
    return array_numeros

def read_from_csv(filename: str) -> Tuple[Tuple[pd.Series, pd.Series]]:
    # Leer el archivo CSV
    df = pd.read_csv(filename)
    epochs = df['epochs']
    runs = df['run']
    df['loss_training'] = df['loss_training'].apply(fix)
    df['loss_validation'] = df['loss_validation'].apply(fix)
    loss_train = df['loss_training']
    loss_val = df['loss_validation']

    return loss_train, loss_val, epochs, runs

def plot_loss(loss_train: pd.Series, loss_val: pd.Series) -> None:
    
    plt.plot(loss_train,  label='Training Loss')
    plt.plot(loss_val, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def write_results() -> None:
    pd.DataFrame([
        {"Case": 1, "Description": "accepted"},
        {"Case": 2, "Description": "underfitting"},
        {"Case": 3, "Description": "overfitting"},
    ]).to_csv("output_ej4.csv", index=False)

if __name__ == "__main__":
    # Llamar a la función con el nombre del archivo CSV
    loss_train, val_train, epochs, runs = read_from_csv("ej4.csv")
    
    for run in runs:
        plot_loss(loss_train[run], val_train[run])
    
    write_results()
