import itertools
import numpy as np

Sigma = np.array([
    [2,1,1,1],
    [1,2,1,1],
    [1,1,2,1],
    [1,1,1,2]
])

w = np.random.uniform(-1,1,size=(1, 4)) #filas #columnas
x_train = np.random.multivariate_normal([0,0,0,0], Sigma)
x_train = x_train.reshape(-1, 1)

print(x_train)
O = np.dot(w, x_train)
print(O*w.T)

Sigma = np.array([
    [2, 1, 1, 1],
    [1, 2, 1, 1],
    [1, 1, 2, 1],
    [1, 1, 1, 2]
])

autovalores, autovectores = np.linalg.eig(Sigma)

print("Autovalores:")
print(autovalores)

print("Autovectores:")
print((autovectores.T[1]).reshape(-1, 1))

print(x_train*x_train)
a = np.zeros((5, 4))
a[0] = [1,2,3,4]
a[1] = [5,6,7,8]
print(a[:,0])
print(w)
print(w.T)


f = True
while f:
    x = np.random.uniform(-1.1,1.1)
    y = np.random.uniform(0,1.1)

    r = np.linalg.norm([x,y])
    print("xy",x,y,r)
    if( 0.9 <= r <= 1.1 and 0 <= np.arctan2(y,x) <= np.pi ):
        f = False

print(r)

# Crear el vector w
w = np.array([[np.random.uniform(-1, 1), 0.25] for _ in range(10)])




print("El vector w generado es:", w)

a = np.array([[1,2],[3,4] ])
b = np.array([[1,2]])
norm = np.linalg.norm(a-b, axis=1)
print(norm)

vacio = plot_w = np.zeros((4, 3, 2))
print(vacio[0][2])

w = np.array([[np.random.uniform(-0.5, 0.5), 0.25] for _ in range(10)])
print(w)
print(w[0])