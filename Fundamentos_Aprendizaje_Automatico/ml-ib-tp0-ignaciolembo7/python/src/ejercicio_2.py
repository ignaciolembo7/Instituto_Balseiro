# Suma de Números
def sumar(a, b):
    s = a + b 
    return s 

# Pitágoras
def pitagoras(a, b, tipo="hipotenusa"):
    if tipo == "hipotenusa":
        return (a**2 + b**2)**(0.5)
    elif tipo == "cateto":
        if a > b:
            return (a**2 - b**2)**(0.5)
        else:
            return (b**2 - a**2)**(0.5)
    else:
        raise ValueError("El tipo debe ser 'hipotenusa' o 'cateto'")