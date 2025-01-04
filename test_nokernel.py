import cupy as cp
import numpy as np
from scipy.constants import m_e
import sys

EARTH_DIPOLE_B0 = -30e3    # nT
MAX_ITERS = 100
EARTH_RADIUS = 6371.1e3   # m

dt = 1                    # s
vpar_start = .1           # earth radii / sec
mu = 700                  # MeV/Gauss
grid_spacing = 0.05
grad_step = 1e-3

# Setup grid
x_axis = cp.arange(-10, 10, grid_spacing)
y_axis = cp.arange(-10, 10, grid_spacing)
z_axis = cp.arange(-5, 5, grid_spacing)

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
Btot = cp.sqrt(Bx**2 + By**2 + Bz**2)

print('B{x,y,z} shape', Bx.shape)

# Instantiate particle positions and parallel velocity
pos_x = cp.arange(3, 6, .01)
pos_y = cp.zeros(pos_x.shape)
pos_z = cp.zeros(pos_x.shape)

print('pos{x,y,z} shape', pos_x.shape)

vpar = vpar_start * cp.ones(pos_x.shape)

print('vpar shape', vpar.shape)

# Loop particles
def interp_field(field, dx=0, dy=0, dz=0):
    field_i = cp.searchsorted(x_axis, pos_x + dx)
    field_j = cp.searchsorted(y_axis, pos_y + dy)
    field_k = cp.searchsorted(z_axis, pos_z + dz)

    result = cp.zeros(pos_x.shape)
    w_accum = cp.zeros(pos_x.shape)
    
    for di in range(2):
        for dj in range(2):
            for dk in range(2):                
                dist_x = x_axis[field_i + di] - (pos_x + dx)
                dist_y = y_axis[field_j + dj] - (pos_y + dy)
                dist_z = z_axis[field_k + dk] - (pos_z + dz)
                                
                w = 1 / cp.sqrt(dist_x**2 + dist_y**2 + dist_z**2)                
                f = field[field_i + di, field_j + dj, field_k + dk]

                result += w * f
                w_accum += w
    
    return result / w_accum

hist_x = []  # history
hist_y = []
hist_z = []
                
for i in range(MAX_ITERS):
    # Get B field
    Bx_cur = interp_field(Bx)
    By_cur = interp_field(By)
    Bz_cur = interp_field(Bz)
    Btot_cur = interp_field(Btot)
    
    # Get Bgrad
    Bgrad_x = (interp_field(Btot, dx=grad_step) - Btot_cur) / grad_step
    Bgrad_y = (interp_field(Btot, dy=grad_step) - Btot_cur) / grad_step
    Bgrad_z = (interp_field(Btot, dz=grad_step) - Btot_cur) / grad_step    

    # Step Position    
    pos_x += vpar * (Bx_cur / Btot_cur) * dt
    pos_y += vpar * (By_cur / Btot_cur) * dt
    pos_z += vpar * (Bz_cur / Btot_cur) * dt

    # Step parallel velocity
    bhat_dot_gradB = (Bx_cur * Bgrad_x + By_cur * Bgrad_y + Bz_cur * Bgrad_z) / Btot_cur
    step = - dt * 2.512e-23 * (mu / m_e) * bhat_dot_gradB

    #print(Bgrad_x)
    #import pdb; pdb.set_trace()
    #vpar += step
                             
    #import pdb
    #pdb.set_trace()
    
    #print(Btot_cur)
    
    
    # Record history
    hist_x.append(pos_x.get())
    hist_y.append(pos_y.get())
    hist_z.append(pos_z.get())

    print('.', end='')
    sys.stdout.flush()

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

    
