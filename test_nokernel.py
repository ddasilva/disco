import cupy as cp
import sys
import time
import pandas as pd
from cupyx import jit
import math

EARTH_DIPOLE_B0 = -30e3   # nT
MAX_ITERS = 5000
EARTH_RADIUS = 6371.1e3   # m

dt = .1                   # s
vpar_start = .1           # earth radii / sec
grid_spacing = 0.1
grad_step = 1e-1          # finite diff delta
mu = 1e-6                 # first invariant


def main():
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
    pos_x = cp.arange(3, 8, .01)
    pos_y = cp.zeros(pos_x.shape)
    pos_z = cp.zeros(pos_x.shape)
    
    print('pos{x,y,z} shape', pos_x.shape)

    vpar = vpar_start * cp.ones(pos_x.shape)
    
    print('vpar shape', vpar.shape)

    # history
    hist_x = cp.zeros((MAX_ITERS, pos_x.size))
    hist_y = cp.zeros((MAX_ITERS, pos_y.size))
    hist_z = cp.zeros((MAX_ITERS, pos_z.size))
    
    start_time = time.time()
    sim(
        pos_x, pos_y, pos_z, vpar,
        hist_x, hist_y, hist_z,
        Bx, By, Bz, Btot,
        x_axis, y_axis, z_axis,
        x_grid, y_grid, z_grid
    )
    end_time = time.time()
    
    print('time = ', end_time - start_time, '( iters = ', MAX_ITERS, ')')

    # Collect history
    hist_x = hist_x.get()
    hist_y = hist_y.get()
    hist_z = hist_z.get()
    
    print('hist_x shape', hist_x.shape)

    # Write output for visualization
    d = {}
    
    for i in range(0, pos_x.size, 50):
        d[f'x{i}'] = hist_x[:, i]
        d[f'y{i}'] = hist_y[:, i]
        d[f'z{i}'] = hist_z[:, i]        
    
    pd.DataFrame(d).to_csv('out.csv')


def sim(
        pos_x, pos_y, pos_z, vpar,
        hist_x, hist_y, hist_z,
        Bx, By, Bz, Btot,
        x_axis, y_axis, z_axis,
        x_grid, y_grid, z_grid
):
    for i in range(MAX_ITERS):
        # Record history
        hist_x[i, :] = pos_x
        hist_y[i, :] = pos_y
        hist_z[i, :] = pos_z
        
        # Get B field
        Bx_cur = interp_field(Bx, pos_x, pos_y, pos_z, x_axis, y_axis, z_axis)
        By_cur = interp_field(By, pos_x, pos_y, pos_z, x_axis, y_axis, z_axis)
        Bz_cur = interp_field(Bz, pos_x, pos_y, pos_z, x_axis, y_axis, z_axis)
        Btot_cur = interp_field(Btot, pos_x, pos_y, pos_z, x_axis, y_axis, z_axis)
    
        # Step Position    
        pos_x += vpar * (Bx_cur / Btot_cur) * dt
        pos_y += vpar * (By_cur / Btot_cur) * dt
        pos_z += vpar * (Bz_cur / Btot_cur) * dt

        # Step parallel velocity
        grad_step_x = grad_step * Bx_cur / Btot_cur
        grad_step_y = grad_step * By_cur / Btot_cur
        grad_step_z = grad_step * Bz_cur / Btot_cur    

        bhat_dot_gradB = (
            interp_field(Btot, pos_x, pos_y, pos_z, x_axis, y_axis, z_axis,
                         dx=grad_step_x, dy=grad_step_y, dz=grad_step_z)
            - Btot_cur
        ) / grad_step
        
        # the constant against bhat dot gradB is fudged for now-- need to work
        # out specific value
        vpar += - dt * mu * bhat_dot_gradB 
    
        print('.', end='')
        sys.stdout.flush()

    print()


def interp_field(
        field,
        pos_x, pos_y, pos_z,
        x_axis, y_axis, z_axis,
        dx=0, dy=0, dz=0
):
    field_i = cp.searchsorted(x_axis, pos_x + dx)
    field_j = cp.searchsorted(y_axis, pos_y + dy)
    field_k = cp.searchsorted(z_axis, pos_z + dz)

    result = cp.zeros(pos_x.shape)
    
    arr_size = pos_x.size
    block_size = 1024
    grid_size = int(math.ceil(arr_size / block_size))
    
    interp_field_kernel[grid_size, block_size](
        arr_size, result, field,
        field_i, field_j, field_k,
        pos_x + dx, pos_y + dy, pos_z + dz,
        x_axis, y_axis, z_axis
    )

    return result
    

@jit.rawkernel()
def interp_field_kernel(
        arr_size, result, field,
        field_i, field_j, field_k,
        pos_x, pos_y, pos_z,
        x_axis, y_axis, z_axis
):
    idx = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x
    
    if idx < arr_size:    
        w_accum = 0.0
    
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    dist_x = x_axis[field_i[idx] + di] - pos_x[idx]
                    dist_y = y_axis[field_j[idx] + dj] - pos_y[idx]
                    dist_z = z_axis[field_k[idx] + dk] - pos_z[idx]
                    
                    w = 1 / cp.sqrt(dist_x**2 + dist_y**2 + dist_z**2)                
                    f = field[field_i[idx] + di, field_j[idx] + dj, field_k[idx] + dk]
                    
                    result[idx] += w * f
                    w_accum += w
    
        result[idx] /= w_accum
    
    
# def interp_field_backup(field, pos_x, pos_y, pos_z, x_axis, y_axis, z_axis, dx=0, dy=0, dz=0):
#     field_i = cp.searchsorted(x_axis, pos_x + dx)
#     field_j = cp.searchsorted(y_axis, pos_y + dy)
#     field_k = cp.searchsorted(z_axis, pos_z + dz)

#     result = cp.zeros(pos_x.shape)
#     w_accum = cp.zeros(pos_x.shape)
    
#     for di in range(2):
#         for dj in range(2):
#             for dk in range(2):
#                 # TODO this requires GPU memory lookups in the
#                 # x_axis[field_i + di] etc terms, and is expensive-- how to do
#                 # this better?
#                 dist_x = x_axis[field_i + di] - (pos_x + dx)
#                 dist_y = y_axis[field_j + dj] - (pos_y + dy)
#                 dist_z = z_axis[field_k + dk] - (pos_z + dz)
                                
#                 w = 1 / cp.sqrt(dist_x**2 + dist_y**2 + dist_z**2)                
#                 f = field[field_i + di, field_j + dj, field_k + dk]

#                 result += w * f
#                 w_accum += w
    
#     return result / w_accum



if __name__ == '__main__':
    main()
