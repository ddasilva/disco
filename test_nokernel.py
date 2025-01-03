import cupy as cp
import numpy as np

EARTH_DIPOLE_B0 = 30e3  # nT
MAX_ITERS = 100
EARTH_RADIUS = 6371.1e3 # m

dt = 1e-2                 # s

# Setup grid
x_axis = cp.arange(-10, 10, .1)
y_axis = cp.arange(-10, 10, .1)
z_axis = cp.arange(-5, 5, .1)

x_grid, y_grid, z_grid = cp.meshgrid(
    x_axis, y_axis, z_axis,
    indexing='ij'
)

print('{x,y,z}_grid Shape', x_grid.shape)

# Calculate Dipole on Grid
r_grid = cp.sqrt(x_grid**2 + y_grid**2 + z_grid**2)

Bx = 3 * x_grid * z_grid * EARTH_DIPOLE_B0 / r_grid**5
By = 3 * y_grid * z_grid * EARTH_DIPOLE_B0 / r_grid**5
Bz = (3 * z_grid**2 - r_grid**2) * EARTH_DIPOLE_B0 / r_grid**5

print('B{x,y,z} shape', Bx.shape)

# Instantiate particle 
pos_x = cp.arange(3, 6, .01)
pos_y = cp.zeros(pos_x.shape)
pos_z = cp.zeros(pos_x.shape)

print('pos{x,y,z} shape', pos_x.shape)

vpar_val = 5.93e7/EARTH_RADIUS # 10 keV elctron, in units of Re/s
vpar = vpar_val * cp.ones(pos_x.shape)

print('vpar shape', vpar.shape)

# Loop particles
def interp_field(field, dx=0, dy=0, dz=0):
    field_i = cp.searchsorted(x_axis, pos_x + dx)
    field_j = cp.searchsorted(y_axis, pos_y + dy)
    field_k = cp.searchsorted(z_axis, pos_z + dz)

    result = cp.zeros(pos_x.shape)
    
    for di in range(2):
        for dj in range(2):
            for dk in range(2):                
                dist_x = x_axis[field_i + di] - pos_x
                dist_y = y_axis[field_j + dj] - pos_y
                dist_z = z_axis[field_k + dk] - pos_z
                                
                w = 1 / cp.sqrt(dist_x**2 + dist_y**2 + dist_z**2)                
                f = field[field_i + di, field_j + dj, field_k + dk]
                result += w * f

    return result


hist_x = []
hist_y = []
hist_z = []
                
for i in range(MAX_ITERS):
    Bx_cur = interp_field(Bx)
    By_cur = interp_field(By)
    Bz_cur = interp_field(Bz)

    Btot_cur = cp.sqrt(Bx_cur**2 + By_cur**2 + Bz_cur**2)
    
    pos_x += vpar * (Bx_cur / Btot_cur) * dt
    pos_y += vpar * (By_cur / Btot_cur) * dt
    pos_z += vpar * (Bz_cur / Btot_cur) * dt

    hist_x.append(pos_x.get())
    hist_y.append(pos_y.get())
    hist_z.append(pos_z.get())

    print('.', end='')

print()
    
hist_x = np.array(hist_x)
hist_y = np.array(hist_y)
hist_z = np.array(hist_z)

print('hist_x shape', hist_x.shape)

# Write output for visualization
d = {}

for i in range(0, pos_x.size, 50):
    d[f'x{i}'] = hist_x[:, i]
    d[f'y{i}'] = hist_y[:, i]
    d[f'z{i}'] = hist_z[:, i]

import pandas as pd
pd.DataFrame(d).to_csv('out.csv')

    
