import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Crear datos de ejemplo
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.2, 100)  # Curva con ruido

# Encontrar los picos en la curva
peaks, _ = find_peaks(y, prominence=0.5)  # Ajusta la prominencia según tu necesidad
print(peaks)
# Graficar los datos y los picos encontrados
plt.plot(x, y, label="Curva con ruido")
plt.plot(x[peaks], y[peaks], "ro", label="Picos")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Identificación de picos en una curva")
plt.show()