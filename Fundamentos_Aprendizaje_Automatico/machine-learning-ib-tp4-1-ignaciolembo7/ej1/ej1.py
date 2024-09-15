# Importar librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# En este caso solo usamos tensorflow por la simplicidad de que ya tiene el dataset incluído en la librería

class feed_forward:

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    # 2. Implementación de Funciones de Activación
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # 3. Función de Pérdida y Métrica de Precisión
    def cross_entropy_loss(self, y_pred, y_true):
        n_samples = y_true.shape[0]
        log_p = -np.log(y_pred[np.arange(n_samples), y_true])
        loss = np.sum(log_p) / n_samples
        return loss

    def accuracy(self, y_pred, y_true):
        y_pred = np.argmax(y_pred, axis=1)
        return np.mean(y_pred == y_true)

    # 4. Forward Propagation
    def forward_propagation(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.softmax(z2)

        return z1, a1, z2, a2
    
    def predict(self, X, y):
        _, _, _, a2 =  self.forward_propagation(X)
        loss_val = self.cross_entropy_loss(a2, y)
        acc_val = self.accuracy(a2, y)

        return loss_val, acc_val, a2

    # 5. Backward Propagation
    def backward_propagation(self, X, y, a1, a2, z1):

        m = y.shape[0]
        one_hot_y = np.zeros((m, self.output_size))
        one_hot_y[np.arange(m), y] = 1
        
        dz2 = a2 - one_hot_y
        dW2 = np.dot(a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        return dW1, db1, dW2, db2

    def train(self, X_train, y_train, epochs, learning_rate):

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            
        np.random.seed(42)
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))

        print("W1 shape: ", self.W1.shape)
        print("b1 shape: ", self.b1.shape)
        print("W2 shape: ", self.W2.shape)
        print("b2 shape: ", self.b2.shape)

        losses_train = []
        accuracies_train = []
        losses_val = []
        accuracies_val = []

        for epoch in tqdm(range(epochs), desc="Training Progress"):
            
            z1_t, a1_t, z2_t, a2_t =  self.forward_propagation(X_train)
            loss_train = self.cross_entropy_loss(a2_t, y_train)
            acc_train = self.accuracy(a2_t, y_train)
            losses_train.append(loss_train)
            accuracies_val.append(acc_train)

            dW1, db1, dW2, db2 = self.backward_propagation(X_train, y_train, a1_t, a2_t, z1_t)

            _, _, _, a2_val =  self.forward_propagation(X_val)
            loss_val = self.cross_entropy_loss(a2_val, y_val)
            acc_val = self.accuracy(a2_val, y_val)
            losses_val.append(loss_val)
            accuracies_val.append(acc_val)

            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2

            if epoch % 25 == 0:
                print(f"Train: Epoch {epoch}: Loss = {loss_train:.2f}, Accuracy = {acc_train:.2f}") 
                print(f"Validation: Epoch {epoch}: Loss = {loss_val:.2f}, Accuracy = {acc_val:.2f}")
            
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(losses_train, label = "Error de entrenamiento")
        plt.plot(losses_val, label = "Error de validación")
        plt.title('Error')
        plt.xlabel('Épocas')	
        plt.ylabel('Error')
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(accuracies_train, label = "Precisión de entrenamiento")
        plt.plot(accuracies_val, label = "Precisión de validación")
        plt.title('Precisión')
        plt.xlabel('Épocas')
        plt.ylabel('Precisión')
        plt.grid()

        plt.tight_layout()
        plt.show()