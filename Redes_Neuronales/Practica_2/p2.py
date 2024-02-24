import numpy as np
import matplotlib.pyplot as plt

#Condiciones iniciales
tau=1
gee=1
gei=0.5
gie=1
gii=1
Ie=1
Ii=3
def S(f):
    f[f<0]=0
    return f

xmin=0
xmax=3
ymin=0
ymax=4
fi, fe = np.mgrid[ymin:ymax:100j, xmin:xmax:100j]
dfe = (-fe+S(gee*fe-gei*fi+Ie))/tau
dfi = (-fi+S(gie*fe-gii*fi+Ii))/tau

fig, ax = plt.subplots()
ax.streamplot(fe,fi,dfe,dfi,density=[3,3])
x=np.linspace(xmin,xmax,100)
ax.plot(x,(gee-1)/gei*x+Ie/gei)
ax.plot(x,gie*x/(gii+1)+Ii/(gii+1))
ax.set_ylim(ymin,ymax)
ax.set_xlim(xmin,xmax)
ax.set_xlabel('he')
ax.set_ylabel('hi')
plt.show()