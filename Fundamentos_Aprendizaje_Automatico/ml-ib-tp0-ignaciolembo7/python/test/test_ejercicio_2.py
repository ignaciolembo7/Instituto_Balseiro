from ..src.ejercicio_2 import sumar, pitagoras

def test_sumar():
    assert sumar(5, 3) == 8
    assert sumar(-1, 1) == 0
    assert sumar(0, 0) == 0

def test_pitagoras():
    assert pitagoras(3, 4, "hipotenusa") == 5
    assert pitagoras(5, 13, "cateto") == 12