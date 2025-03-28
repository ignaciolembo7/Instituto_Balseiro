
# Ejercicio 3: Clases

# Rectángulo
class Rectangulo:
    def __init__(self, longitud, ancho):
        self.longitud = longitud
        self.ancho = ancho

    def area(self):
        return self.longitud * self.ancho

    def perimetro(self):
        return 2 * (self.longitud + self.ancho)

# Cuadrado
class Cuadrado(Rectangulo):
    def __init__(self, lado):
        super().__init__(lado, lado) 

