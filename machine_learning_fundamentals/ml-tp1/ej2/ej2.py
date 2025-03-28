import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

def read_from_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)

def confusion_matrix(y_true, y_pred):
    cm = sklearn_confusion_matrix(y_true, y_pred)
    #plt.figure(figsize=(8, 6))
    #sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    #plt.xlabel('Predicted Labels')
    #plt.ylabel('True Labels')
    #plt.title('Confusion Matrix')
    #plt.show()
    return np.asarray(cm)

if __name__ == "__main__":
    # Load data from the CSV file
    dataframe = read_from_csv("ej2_a.csv")
    y_true = dataframe['y']
    y_pred = dataframe['y_pred']

    # Calculate and plot the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(cm)