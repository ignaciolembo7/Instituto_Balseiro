from ..src.ejercicio_3 import Rectangulo, Cuadrado

def test_rectangulo():
    rect = Rectangulo(4, 3)
    assert rect.area() == 12
    assert rect.perimetro() == 14

def test_cuadrado():
    cuad = Cuadrado(5)
    assert cuad.area() == 25
    assert cuad.perimetro() == 20