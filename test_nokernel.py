import cupy as cp
import sys
import time
import pandas as pd
from cupyx import jit
import math
from scipy.constants import elementary_charge, m_e
from dataclasses import dataclass
from typing import Any
#import line_profiler

EARTH_DIPOLE_B0 = -30e3   # nT
MAX_ITERS = 5000

dt = .1                   # s
vpar_start = .1           # earth radii / sec
grid_spacing = 0.1
grad_step = 5e-2          # finite diff delta step (half)
mu = 1e-36                # first invariant


@dataclass
class Positions:
    """1D arrays of cartesian particle position component"""
    x: Any
    y: Any
    z: Any

@dataclass
class Axes:
    """1D arrays of rectilinear grid axes"""
    x: Any
    y: Any
    z: Any    

@dataclass
class PositionHistory:
    """History of positions"""
    x: Any
    y: Any
    z: Any


@dataclass
class Neighbors:
    """Neighbors of given particles, used for interpolation"""
    field_i: Any
    field_j: Any
    field_k: Any
    
    
def main():
    """Main method of the program."""
    # Setup axes and grid
    x_axis = cp.arange(-10, 10, grid_spacing)
    y_axis = cp.arange(-10, 10, grid_spacing)
    z_axis = cp.arange(-5, 5, grid_spacing)
    axes = Axes(x_axis, y_axis, z_axis)
    
    x_grid, y_grid, z_grid = cp.meshgrid(
        axes.x, axes.y, axes.z,
        indexing='ij'
    )

    print('Grid Shape:', x_grid.shape)
    
    # Calculate Dipole on Grid
    r_grid = cp.sqrt(x_grid**2 + y_grid**2 + z_grid**2)
    
    Bx = 3 * x_grid * z_grid * EARTH_DIPOLE_B0 / r_grid**5
    By = 3 * y_grid * z_grid * EARTH_DIPOLE_B0 / r_grid**5
    Bz = (3 * z_grid**2 - r_grid**2) * EARTH_DIPOLE_B0 / r_grid**5
    Btot = cp.sqrt(Bx**2 + By**2 + Bz**2)
        
    # Instantiate particle positions and parallel velocity
    pos_x = cp.arange(3, 8, .01)
    pos_y = cp.zeros(pos_x.shape)
    pos_z = cp.zeros(pos_x.shape)

    pos = Positions(pos_x, pos_y, pos_z)
    print('Number of particles:', pos_x.size)

    vpar = vpar_start * cp.ones(pos.x.shape)    

    # History of positions
    hist = PositionHistory(
        x=cp.zeros((MAX_ITERS, pos.x.size)),
        y=cp.zeros((MAX_ITERS, pos.y.size)),
        z=cp.zeros((MAX_ITERS, pos.y.size))
    )

    print('Iterations:', MAX_ITERS)
    
    start_time = time.time()
    sim(pos, vpar, hist, Bx, By, Bz, Btot, axes)
    end_time = time.time()
    
    print('time = ', end_time - start_time, 's')

    # Collect history
    hist_x = hist.x.get()
    hist_y = hist.y.get()
    hist_z = hist.z.get()

    # Write output for visualization
    d = {}
    
    for i in range(0, pos.x.size, 50):
        d[f'x{i}'] = hist_x[:, i]
        d[f'y{i}'] = hist_y[:, i]
        d[f'z{i}'] = hist_z[:, i]        
    
    pd.DataFrame(d).to_csv('out.csv')

    
#@line_profiler.profile
def sim(pos, vpar, hist, Bx, By, Bz, Btot, axes):
    for i in range(MAX_ITERS):
        # Record history of positions
        hist.x[i, :] = pos.x
        hist.y[i, :] = pos.y
        hist.z[i, :] = pos.z
        
        # Get B and Bhat vectors at particle locations
        Bx_cur, neighbors = interp_field(Bx, pos, axes)
        By_cur, _ = interp_field(By, pos, axes, neighbors=neighbors)
        Bz_cur, _ = interp_field(Bz, pos, axes, neighbors=neighbors)
        Btot_cur, _ = interp_field(Btot, pos, axes, neighbors=neighbors)

        Bx_hat_cur = Bx_cur / Btot_cur
        By_hat_cur = By_cur / Btot_cur
        Bz_hat_cur = Bz_cur / Btot_cur
        
        # Step Position
        # -----------------------------------------
        # gradB guiding center term
        gradB_x = (
            interp_field(Btot, pos, axes, dx=grad_step)[0]
            - interp_field(Btot, pos, axes, dx=-grad_step)[0]
        ) / (2 * grad_step)
        gradB_y = (
            interp_field(Btot, pos, axes, dy=grad_step)[0]
            - interp_field(Btot, pos, axes, dy=-grad_step)[0]
        ) / (2 * grad_step)
        gradB_z = (
            interp_field(Btot, pos, axes, dz=grad_step)[0]
            - interp_field(Btot, pos, axes, dz=-grad_step)[0]
        ) / (2 * grad_step)

        q = -elementary_charge

        cross_term1 = cp.zeros((Bx_cur.size, 3))
        cross_term1[:, 0] = Bx_hat_cur
        cross_term1[:, 1] = By_hat_cur
        cross_term1[:, 2] = Bz_hat_cur
      
        cross_term2 = cp.zeros((Bx_cur.size, 3))
        cross_term2[:, 0] = gradB_x
        cross_term2[:, 1] = gradB_y
        cross_term2[:, 2] = gradB_z 

        v_perp = cp.sqrt(mu * Btot_cur  / m_e) 
        scale_term = m_e * v_perp**2  / (2 * q * Btot_cur) 

        scale_term *= 1e12        # TODO need to work out the units
        
        v_gradB = cp.zeros((Bx_cur.size, 3))
        v_gradB[:, :] = cp.cross(cross_term1, cross_term2)

        for i in range(3):
            v_gradB[:, i] *= scale_term
        
        # do the step
        pos.x += dt * (vpar * Bx_hat_cur + v_gradB[:, 0])
        pos.y += dt * (vpar * By_hat_cur + v_gradB[:, 1])
        pos.z += dt * (vpar * Bz_hat_cur + v_gradB[:, 2])

        # Step parallel velocity
        # -----------------------------------
        # bhat dot gradB
        grad_step_x = grad_step * Bx_hat_cur
        grad_step_y = grad_step * By_hat_cur 
        grad_step_z = grad_step * Bz_hat_cur
        
        bhat_dot_gradB = (
            interp_field(
                Btot, pos, axes, 
                dx=grad_step_x, dy=grad_step_y, dz=grad_step_z
            )[0]
            -
            interp_field(
                Btot, pos, axes, 
                dx=-grad_step_x, dy=-grad_step_y, dz=-grad_step_z
            )[0]         
        ) / (2 * grad_step)
        
        # do the step
        vpar += - dt * (mu / m_e) * bhat_dot_gradB 
        
        print('.', end='')
        sys.stdout.flush()

    print()


#@line_profiler.profile
def interp_field(field, pos, axes, neighbors=None, dx=0, dy=0, dz=0):
    """Interpolate a 3D gridded field at given positions.

    Args
      field: 3D gridded field to interpolate
      pos: position vector
      axes: rectilinear grid axes
      neighbors: optional, reuse this value for lookup of the neighbors
      dx, dy, dz: optional, perturb the position by these values    
    Return
      result: cupy array of interpolated field values at position
      neighbors: neighbors object for reuse
    """ 
    pos_x_pert = pos.x + dx
    pos_y_pert = pos.y + dy
    pos_z_pert = pos.z + dz
    
    if neighbors is None:
        neighbors = Neighbors(
            field_i=cp.searchsorted(axes.x, pos_x_pert),
            field_j=cp.searchsorted(axes.y, pos_y_pert),
            field_k=cp.searchsorted(axes.z, pos_z_pert),
        )
        
    result = cp.zeros(pos.x.shape)    
    arr_size = pos.x.size
    block_size = 1024
    grid_size = int(math.ceil(arr_size / block_size))
    
    interp_field_kernel[grid_size, block_size](
        arr_size, result, field,
        neighbors.field_i, neighbors.field_j, neighbors.field_k,
        pos_x_pert, pos_y_pert, pos_z_pert,
        axes.x, axes.y, axes.z
    )

    return result, neighbors
    

@jit.rawkernel()
def interp_field_kernel(
        arr_size, result, field,
        field_i, field_j, field_k,
        pos_x, pos_y, pos_z,
        x_axis, y_axis, z_axis
):
    """[CUPY KERNEL] Interpolate field using neighbors.
    
    Uses inverse distance weighted average. This kernel operates
    on one position per thread.
    """
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


if __name__ == '__main__':
    main()
