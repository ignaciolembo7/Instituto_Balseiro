import numpy as np

def rk4(funcs, y0, t0, dt):
    k1 = np.array([func(t0, y0) for func in funcs])
    k2 = np.array([func(t0 + dt/2, y0 + dt/2 * k1) for func in funcs])
    k3 = np.array([func(t0 + dt/2, y0 + dt/2 * k2) for func in funcs])
    k4 = np.array([func(t0 + dt, y0 + dt * k3) for func in funcs])
    
    y_new = y0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y_new
