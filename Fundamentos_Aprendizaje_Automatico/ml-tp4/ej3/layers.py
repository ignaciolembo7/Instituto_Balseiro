class BaseLayer:
    def forward(self, X):
        pass  # Completar

    def backward(self, grad):
        pass  # Completar

class Input(BaseLayer):
    def __init__(self, shape):
        pass  # Completar

    def forward(self, X):
        pass  # Completar

    def backward(self, grad):
        pass  # Completar

class Layer(BaseLayer):
    def __init__(self):
        pass  # Completar

class Dense(Layer):
    def __init__(self, input_size, output_size, activation):
        pass  # Completar

    def forward(self, X):
        pass  # Completar

    def backward(self, grad):
        pass  # Completar
