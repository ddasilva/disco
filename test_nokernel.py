import time
import cupy as cp
import pandas as pd

import libgputrace 

from scipy.constants import m_e


EARTH_DIPOLE_B0 = -30e3   # nT

    
    
def main():
    """Main method of the program."""
    config = libgputrace.TraceConfig(
        max_iters=5000,          # number of iterations
        dt=.1,                   # s
        vpar_start=.1,           # earth radii / sec
        mu=1e-36,                # first invariant, 1e9 (R_earth**2) A
        particle_mass=m_e,
    )
    grid_spacing = 0.1 
    
    # Setup axes and grid
    x_axis = cp.arange(-10, 10, grid_spacing)
    y_axis = cp.arange(-10, 10, grid_spacing)
    z_axis = cp.arange(-5, 5, grid_spacing)
    axes = libgputrace.Axes(x_axis, y_axis, z_axis)
    
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

    Ex = cp.zeros(Bx.shape)
    Ey = cp.zeros(Bx.shape)
    Ez = cp.zeros(Bx.shape)
    
    # Instantiate particle positions and parallel velocity
    pos_x = cp.arange(3, 8, .01)
    pos_y = cp.zeros(pos_x.shape)
    pos_z = cp.zeros(pos_x.shape)

    pos = libgputrace.Positions(pos_x, pos_y, pos_z)
    print('Number of particles:', pos_x.size)

    vpar = config.vpar_start * cp.ones(pos.x.shape)    

    # History of positions
    hist = libgputrace.PositionHistory(
        x=cp.zeros((config.max_iters, pos.x.size)),
        y=cp.zeros((config.max_iters, pos.y.size)),
        z=cp.zeros((config.max_iters, pos.y.size))
    )

    print('Iterations:', config.max_iters)
    
    start_time = time.time()
    libgputrace.trace_trajectory(config, pos, vpar, hist, Bx, By, Bz, Btot, Ex, Ey, Ez, axes)
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


if __name__ == '__main__':
    main()
