import pandas as pd
from typing import Tuple
import numpy as np

def read_from_csv(filename: str) -> Tuple[pd.Series, pd.Series]:
    """Lee los datos desde un archivo CSV y devuelve dos series."""
    data = pd.read_csv(filename)
    y_true = data['y']
    y_pred = data['y_pred']
    return y_true, y_pred

def precision(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calcula la precisión."""
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    return true_positive / (true_positive + false_positive)

def recall(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calcula la exhaustividad."""
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    return true_positive / (true_positive + false_negative)

def f1_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calcula la puntuación F1."""
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec)

def mse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calcula el error cuadrático medio."""
    return np.mean((y_true - y_pred) ** 2)

def cross_entropy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calcula la entropía cruzada."""
    epsilon = 1e-8  # Valor pequeño para evitar divisiones por cero en el logaritmo
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Se asegura de que los valores estén en el rango [epsilon, 1 - epsilon]
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

if __name__ == "__main__":
    # Leer los datos desde el archivo CSV
    y_true, y_pred = read_from_csv("ej1.csv")

    # Calcular y mostrar métricas
    print("Precision:", precision(y_true, y_pred))
    print("Recall:", recall(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("MSE:", mse(y_true, y_pred))
    print("Cross Entropy:", cross_entropy(y_true, y_pred))