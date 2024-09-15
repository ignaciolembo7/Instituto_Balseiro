class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        pass  # Completar

    def compile(self, loss, optimizer):
        pass  # Completar

    def forward(self, X):
        pass  # Completar

    def backward(self, X, y):
        pass  # Completar

    def fit(self, X, y, epochs, batch_size):
        pass  # Completar

    def predict(self, X):
        pass  # Completar
