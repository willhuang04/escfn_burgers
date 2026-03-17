import numpy as np
import matplotlib.pyplot as plt

n_instances = 50
n_t = 100
n_x = 256

uL = np.random.uniform(0.8, 1.2, size=(n_instances, 1, 1))
uR = np.random.uniform(-0.2, 0.2, size=(n_instances, 1, 1))

xs = np.linspace(-5,5,n_x)
ts = np.linspace(0,5,n_t)

X, T = np.meshgrid(xs, ts)

X_3d = X[np.newaxis, :, :]
T_3d = T[np.newaxis, :, :]

def u(x,t,uL,uR):
    conditions = [
    x < uL**2 * t,
    (x >= uL**2 * t) & (x < uR**2 * t),
    x >= uR**2 * t
    ]
    choices = [
    uL, 
    np.sqrt(x/t), 
    uR
    ]
    return np.select(conditions, choices)
    
Z = u(X, T, uL, uR)
Z = np.expand_dims(Z, axis=-1)

print(np.shape(Z))

np.save('valData_CubicFlux_Expansion_256.npy', Z)
